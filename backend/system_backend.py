# importing_Libraries
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from typing import List
from pydantic import BaseModel
from datetime import datetime, timezone
from fastapi import UploadFile, File
import os
from src.lpr import get_lpr_pipeline
from src.fire_detection.inference import FireDetector
from src.tts.tts_engine import generate_alert_audio   # now just speaks, returns nothing
from src.weapon_detect.predict import WeaponDetection

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentry Vision — AI Security System")

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve dashboard ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if not os.path.exists(dashboard_path):
        raise HTTPException(status_code=404, detail="dashboard.html not found next to system_backend.py")
    with open(dashboard_path, "r", encoding="utf-8") as f:
        return f.read()

# ── Database ───────────────────────────────────────────────────────────────────
engine       = create_engine("sqlite:///./users.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

class PredictionDB(Base):
    __tablename__ = "predictions"
    id                   = Column(Integer, primary_key=True, index=True)
    license_plate_number = Column(String, unique=True, index=True)
    confidence           = Column(Float)
    timestamp            = Column(DateTime)

class Plate(Base):
    __tablename__ = "license_plate"
    license_plate_number = Column(String, primary_key=True, index=True, unique=True)
    timestamp            = Column(DateTime, default=lambda: datetime.now(timezone.utc))

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"DB init warning: {e}")

# ── Pydantic schemas ───────────────────────────────────────────────────────────
class CreatePlate(BaseModel):
    license_plate_number: str

class PlateResponse(BaseModel):
    license_plate_number: str
    timestamp: datetime
    class Config:
        from_attributes = True

class Predict(BaseModel):
    license_plate_number: str
    confidence: float
    timestamp: datetime

class PredictResponse(BaseModel):
    id: int
    license_plate_number: str
    confidence: float
    timestamp: datetime
    class Config:
        from_attributes = True

# ── Startup ────────────────────────────────────────────────────────────────────
lpr_pipeline    = None
fire_detector   = None
weapon_detector = None
MAX_UPLOAD_SIZE = 5 * 1024 * 1024

@app.on_event("startup")
def startup_event():
    global lpr_pipeline, fire_detector, weapon_detector
    lpr_pipeline = get_lpr_pipeline()
    fire_model_path = os.path.join("models", "fire_model", "fire_model.pth")
    if not os.path.exists(fire_model_path):
        fire_model_path = "fire_model.pth"
    fire_detector   = FireDetector(model_path=fire_model_path)
    weapon_detector = WeaponDetection()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── Watchlist endpoints ────────────────────────────────────────────────────────

@app.get("/plates", response_model=List[PlateResponse])
def get_all_plates(db: Session = Depends(get_db)):
    return db.query(Plate).all()

@app.get("/plates/{license_plate_number}", response_model=PlateResponse)
def get_plate(license_plate_number: str, db: Session = Depends(get_db)):
    plate = db.query(Plate).filter(Plate.license_plate_number == license_plate_number).first()
    if plate is None:
        raise HTTPException(status_code=404, detail="Plate Not Found!")
    return plate

@app.post("/plates", response_model=PlateResponse)
def create_plate(plate_data: CreatePlate, db: Session = Depends(get_db)):
    if db.query(Plate).filter(Plate.license_plate_number == plate_data.license_plate_number).first():
        raise HTTPException(status_code=400, detail="Plate already recorded")
    new_plate = Plate(license_plate_number=plate_data.license_plate_number)
    db.add(new_plate); db.commit(); db.refresh(new_plate)
    return new_plate

@app.put("/plates/{license_plate_number}", response_model=PlateResponse)
def update_plate(license_plate_number: str, plate_data: CreatePlate, db: Session = Depends(get_db)):
    db_plate = db.query(Plate).filter(Plate.license_plate_number == license_plate_number).first()
    if db_plate is None:
        raise HTTPException(status_code=404, detail="Plate Not Found!")
    db_plate.license_plate_number = plate_data.license_plate_number
    db_plate.timestamp = datetime.now(timezone.utc)
    db.commit(); db.refresh(db_plate)
    return db_plate

@app.delete("/plates/{license_plate_number}", response_model=PlateResponse)
def delete_plate(license_plate_number: str, db: Session = Depends(get_db)):
    db_plate = db.query(Plate).filter(Plate.license_plate_number == license_plate_number).first()
    if db_plate is None:
        raise HTTPException(status_code=404, detail="Plate Not Found!")
    db.delete(db_plate); db.commit()
    return db_plate

@app.get("/search/{license_plate_number}", response_model=List[PlateResponse])
def search_plates(license_plate_number: str, db: Session = Depends(get_db)):
    return db.query(Plate).filter(Plate.license_plate_number.contains(license_plate_number)).all()

# ── LPR endpoint ───────────────────────────────────────────────────────────────

@app.post("/predict")
async def license_plate_predict(image: UploadFile = File(...), db: Session = Depends(get_db)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        image_bytes = await image.read()
        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Image too large. Max 5MB.")

        results = lpr_pipeline.process_image(image_bytes)

        stolen_detected = False
        for plate in results.get("plates", []):
            # Normalize: strip spaces + uppercase on both detected plate and DB value
            plate_num = plate.get("text", "").replace(" ", "").upper()
            if plate_num:
                db_plate = db.query(Plate).filter(
                    func.upper(func.replace(Plate.license_plate_number, " ", "")) == plate_num
                ).first()
                if db_plate:
                    stolen_detected = True
                    break

        if stolen_detected:
            message = "A Stolen Car has been detected! Please call police immediately "*4
            generate_alert_audio(message)   # speaks on server, returns nothing
            return {"results": results, "stolen_detected": True,  "message": message}
        else:
            return {"results": results, "stolen_detected": False, "message": "No watchlisted plates detected"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LPR error: {str(e)}")


@app.post("/save_predict", response_model=PredictResponse)
def save_predict(predict_data: Predict, db: Session = Depends(get_db)):
    if db.query(PredictionDB).filter(PredictionDB.license_plate_number == predict_data.license_plate_number).first():
        raise HTTPException(status_code=400, detail="Prediction already recorded")
    new = PredictionDB(
        license_plate_number=predict_data.license_plate_number,
        confidence=predict_data.confidence,
        timestamp=predict_data.timestamp,
    )
    db.add(new); db.commit(); db.refresh(new)
    return new

# ── Fire endpoint ──────────────────────────────────────────────────────────────

@app.post("/fire_predict")
async def fire_predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        image_bytes = await image.read()
        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Image too large. Max 5MB.")

        result = fire_detector.detect_fire(image_bytes)

        if result == 1:
            message = "Fire detected! Please call emergency services immediately "*4
            generate_alert_audio(message)   # speaks on server, returns nothing
            return {"fire_detected": True,  "message": message}
        else:
            return {"fire_detected": False, "message": "No fire detected"}

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fire detection error: {str(e)}")

# ── Weapon endpoint ────────────────────────────────────────────────────────────

@app.post("/predict_weapon")
async def weapon_predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        image_bytes = await image.read()
        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Image too large. Max 5MB.")

        result = weapon_detector.process_weapon(image_bytes)

        if result and len(result) > 0:
            message = "Weapon detected! Please call Police immediately "*4
            generate_alert_audio(message)   # speaks on server, returns nothing
            return {"weapon_detected": True,  "message": message}
        else:
            return {"weapon_detected": False, "message": "No weapon detected"}

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weapon detection error: {str(e)}")

# ── Image serve endpoint ───────────────────────────────────────────────────────

@app.get("/Image/{image_name}")
async def get_image(image_name: str):
    image_path = f"images/{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)