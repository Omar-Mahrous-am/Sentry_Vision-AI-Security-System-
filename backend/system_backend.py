#importing_Libraries
from fastapi import FastAPI,HTTPException,status,Path,Depends
from sqlalchemy import create_engine,Column,Integer,String,DateTime,Float
from sqlalchemy.orm import sessionmaker,declarative_base     
from typing import Optional,List
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from fastapi import UploadFile, File
import os
from src.lpr import get_lpr_pipeline
from model import detect_fire
from tts import generate_alert_audio    

#FastAPI_Object
app=FastAPI(title="Integration With SQLITE ")


#Setting_Engine_and_Session
engine=create_engine("sqlite:///./users.db",connect_args={"check_same_thread":False})
SessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base=declarative_base() 

class PredictionDB(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    license_plate_number = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime)



class Predict(BaseModel):
    License_Plate_number: str
    confidence: float   
    timestamp: datetime 

class PredictResponse(BaseModel):
    id: int
    license_plate_number: str
    confidence: float
    timestamp: datetime

    class Config:
        from_attributes = True



#DataBase_Class
class Plate(Base):
    __tablename__="License_Plate"
    License_Plate_number=Column(String,primary_key=True,index=True,unique=True)
    timestamp=Column(DateTime,default=datetime.utcnow)





#DataBase_create_user_Class
class CreatePlate(BaseModel):
    License_Plate_number: str

# DataBase_return_user_Class
class PlateResponse(BaseModel):
    License_Plate_number: str      
    timestamp: datetime
    
    class Config:
        from_attributes=True    


Base.metadata.create_all(bind=engine)



def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()  



#Endpoints


#Create_user_in_DB
@app.get("/")
def root():
    return {"message": "FastAPI with SQL for License Plates"}

@app.get("/plates/{License_Plate_number}",response_model=PlateResponse) 
def get_plate(License_Plate_number: str,db:Session=Depends(get_db)):
    plate=db.query(Plate).filter(Plate.License_Plate_number==License_Plate_number).first()
    if plate is None:
        raise HTTPException(status_code=404,detail="Plate Not Found!!")
    return plate



#Read_user_From_DB
@app.post("/plates/", response_model=PlateResponse) 
def create_plate(plate_data: CreatePlate, db: Session = Depends(get_db)):
    db_plate = db.query(Plate).filter(Plate.License_Plate_number == plate_data.License_Plate_number).first()
    if db_plate:
        raise HTTPException(status_code=400, detail="Plate already recorded")
    
    new_plate = Plate(License_Plate_number=plate_data.License_Plate_number)
    db.add(new_plate)
    db.commit()
    db.refresh(new_plate)
    return new_plate



#Update_newuser_in_DB
@app.put("/plates/{License_Plate_number}",response_model=PlateResponse) 
def update_plate(License_Plate_number: str, plate_data: CreatePlate, db: Session = Depends(get_db)):
    db_plate = db.query(Plate).filter(Plate.License_Plate_number == License_Plate_number).first()
    if db_plate is None:
        raise HTTPException(status_code=404, detail="Plate Not Found!!")
    
    # In this case, we might only update the timestamp or the number itself (if corrected)
    db_plate.License_Plate_number = plate_data.License_Plate_number
    db_plate.timestamp = datetime.utcnow()
    
    db.commit()
    db.refresh(db_plate)
    return db_plate

#Delete_user_from_DB
@app.delete("/plates/{License_Plate_number}",response_model=PlateResponse) 
def delete_plate(License_Plate_number: str, db: Session = Depends(get_db)):
    db_plate = db.query(Plate).filter(Plate.License_Plate_number == License_Plate_number).first()
    if db_plate is None:
        raise HTTPException(status_code=404, detail="Plate Not Found!!")
    db.delete(db_plate)
    db.commit()
    return db_plate


#Searching_by_name_in_DB
@app.get("/search/{License_Plate_number}",response_model=List[PlateResponse]) 
def search_plates(License_Plate_number: str, db: Session = Depends(get_db)):
    plates = db.query(Plate).filter(Plate.License_Plate_number.contains(License_Plate_number)).all()
    return plates    

#Post_predict
@app.post("/predict")
async def License_plate_predict(image:UploadFile=File(...),db:Session=Depends(get_db)):
    """
    Endpoint to predict License Plates in the uploaded image.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        image_bytes = await image.read()
        pipeline = get_lpr_pipeline()
        results = pipeline.process_image(image_bytes)
        
        stolen_detected = False
        # results is likely a list or contains a list of plates
        # We check if any of the detected plates are in our 'tracked' database
        for result in results.get("detections", []):
            plate_num = result.get("plate_number")
            if plate_num:
                db_plate = db.query(Plate).filter(Plate.License_Plate_number == plate_num).first()
                if db_plate:
                    stolen_detected = True
                    break

        if stolen_detected:
            message = "A Stolen or Watchlisted Car has been detected! Please call police immediately. " * 4
            
            # Generate TTS audio
            audio_path = generate_alert_audio(message)
            
            # Since audio could fail, handle it gracefully
            if not audio_path or not os.path.exists(audio_path):
                audio_path = None
                
            return {
                "results": results,
                "stolen_detected": True,
                "message": message,
                "audio_file": audio_path
            }
        else:
            return {
                "results": results,
                "stolen_detected": False,
                "message": "No watchlisted plates detected",
                "audio_file": None
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error processing LPR: {str(e)}")
    

#post_fire_predict
@app.post("/fire_predict")
async  def fire_predict(image:UploadFile=File(...),db:Session=Depends(get_db)):
    """
    Endpoint to predict fire in the uploaded image.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        image_bytes = await image.read()
        result = detect_fire(image_bytes)
        
        if result == 1:
            message = "Fire detected !! Please call emergency services immediately. " *4
            
            # Generate TTS audio
            audio_path = generate_alert_audio(message)
            
            # Since audio could fail, handle it gracefully
            if not audio_path or not os.path.exists(audio_path):
                audio_path = None
                
            return {
                "fire_detected": True,
                "message": message,
                "audio_file": audio_path
            }
        else:
            return {
                "fire_detected": False,
                "message": "No fire detected",
                "audio_file": None
            }
            
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 



@app.post("/save_predict", response_model=PredictResponse)
def save_predict(predict_data: Predict, db: Session = Depends(get_db)):
    """
    Endpoint to save the prediction in the database.
    """
    db_predict = db.query(PredictionDB).filter(PredictionDB.license_plate_number == predict_data.License_Plate_number).first()
    if db_predict:
        raise HTTPException(status_code=400, detail="Prediction already recorded")
    
    new_predict = PredictionDB(
        license_plate_number=predict_data.License_Plate_number,
        confidence=predict_data.confidence,
        timestamp=predict_data.timestamp
    )
    db.add(new_predict)
    db.commit()
    db.refresh(new_predict)
    return new_predict