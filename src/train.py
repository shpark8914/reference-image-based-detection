import time, torch

def train_model(Train_Valid_DataLoader_Set, model, criterion1, criterion2,
                optimizer, scheduler, args, device, total_loss=True, control=0.1, num_epochs=25):
    
    since = time.time()
    
    history = {"train_loss": list(),"valid_loss": list(),"train_acc": list(),"valid_acc": list()}
    history2 = {"CE_Train_Loss" : list(), "CE_Valid_Loss" : list()}
    history3 = {"Second_Train_Loss" : list(), "Second_Valid_Loss" : list()}
    
    epoch_train_time = []
    epoch_valid_time = []


    
    for epoch in range(num_epochs):
        
        epoch_start = time.time()
        print('-' * 10)
        print("Epoch: %d/%d" % (epoch + 1, num_epochs))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate m
            
            
            epoch_loss = 0.0
            epoch_loss1 = 0.0
            epoch_loss2 = 0.0

            epoch_acc = 0
            epoch_examples = 0
            # Iterate over data.
            
            for i, (current_img, master_img, _, label) in enumerate(Train_Valid_DataLoader_Set[phase]):
                
                current_input = current_img.to(device)
                master_input = master_img.to(device)
                target = label.to(device)
                
                label_weight = (1-target).clone()               
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    if args.model == "proposed":
                        output, focused_current_f_map, focused_master_f_map = model(current_input, master_input)
                        if (total_loss) and (phase=='train') and criterion2 != None:
                            loss1 = criterion1(output, target)
                            loss2 = criterion2(focused_current_f_map, focused_master_f_map, label_weight)
                            loss2 *= control
                            loss = loss1 + loss2
                            
                        else:
                            loss = criterion1(output, target)
       
                    else:
                        output, _ = model(current_input)
                        loss = criterion1(output, target)

                    _, predicted = torch.max(output.data, 1)

                     # compute gradient and do SGD step
                    if phase=='train':
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()    
                    
                # performance statistics
                # calculate sum of loss
                epoch_examples += target.size(0)
                
                if phase=='train' and args.model == "proposed" and criterion2 != None:
                    epoch_loss1 += loss1.item()*target.size(0)
                    epoch_loss2 += loss2.item()*target.size(0)
                    epoch_loss += loss.item()*target.size(0)
                else:
                    epoch_loss += loss.item()*target.size(0)

                epoch_acc += (predicted == target).sum().item()
            
                del current_input, master_input, target, output, predicted
            
            # acc and loss per_sample_at each epoch
            loss_per_sample_epoch_ = epoch_loss/epoch_examples
            acc_per_sample_epoch = epoch_acc/epoch_examples
            epoch_end = time.time()
            epoch_time =  epoch_end - epoch_start
            
            # save & print loss and acc
            if phase == 'train':
                if args.model == "proposed" and criterion2 != None:
                    CEloss_per_sample_epoch_ = epoch_loss1/epoch_examples
                    WSEloss_per_sample_epoch_ = epoch_loss2/epoch_examples
                    
                    scheduler.step()
                    history["train_loss"].append(loss_per_sample_epoch_)
                    history["train_acc"].append(acc_per_sample_epoch)
                    history2["CE_Train_Loss"].append(CEloss_per_sample_epoch_)
                    history3["Second_Train_Loss"].append(WSEloss_per_sample_epoch_)
                    epoch_train_time.append(epoch_time)
                    print("Train - Loss: %.6f (%.6f + %.6f), Accuracy: %.2f, Time: %.2f" % (loss_per_sample_epoch_, CEloss_per_sample_epoch_, WSEloss_per_sample_epoch_, acc_per_sample_epoch, epoch_time))
                
                else :
                    scheduler.step()
                    history["train_loss"].append(loss_per_sample_epoch_)
                    history["train_acc"].append(acc_per_sample_epoch)
                    epoch_train_time.append(epoch_time)
                    print("Train - Loss: %.6f , Accuracy: %.2f, Time: %.2f" % (loss_per_sample_epoch_, acc_per_sample_epoch, epoch_time))
                    
            else:
                history["valid_loss"].append(loss_per_sample_epoch_)
                history["valid_acc"].append(acc_per_sample_epoch)
                epoch_valid_time.append(epoch_time)
                print("Valid - Loss: %.6f , Accuracy: %.2f, Time: %.2f" % (loss_per_sample_epoch_, acc_per_sample_epoch, epoch_time))
                
        
        
                
    end = time.time()
    epoch_time = end - since

    
    return history, history2, history3, epoch_time, epoch_train_time, epoch_valid_time