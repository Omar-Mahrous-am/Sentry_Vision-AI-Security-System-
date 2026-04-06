import torch


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_path='best_accuracy_model.pth', epochs=22, patience=5):
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'  loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f}')
        print(f'  val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}')
        
        # Track best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
            }, save_path)
            print(f'  Best validation accuracy improved to {val_accuracy:.4f}')
        
        # Early stopping check based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f'  Validation loss improved')
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter} epoch(s)')
            
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                print(f'Best validation loss: {best_val_loss:.4f}')
                print(f'Best validation accuracy: {best_val_accuracy:.4f}')
                break
    
    return history, best_val_accuracy