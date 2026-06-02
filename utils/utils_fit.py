import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)

            #----------------------#
            #   计算损失
            #----------------------#
            loss_value = yolo_loss(outputs, targets)

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = yolo_loss(outputs, targets)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train_eval(images)

            #----------------------#
            #   计算损失
            #----------------------#
            loss_value = yolo_loss(outputs, targets)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))


# def fit_one_epoch2(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
#     loss        = 0
#     val_loss    = 0

#     if local_rank == 0:
#         print('Start Train')
#         pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
#     model_train.train()
#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break

#         # 多模态：batch[0] 是一个元组 (ct_images, pet_images), batch[1] 是 bboxes
#         # images, bboxes = batch
#         # print(bboxes)
#         # ct_images, pet_images = images  # 解包得到CT和PET图像
#         images, bboxes = batch[0], batch[1]
#         # print(len(images))
#         ct_images, pet_images = images 
#         # print(ct_images.size(), pet_images.size())
#         with torch.no_grad():
#             if cuda:
#                 ct_images = ct_images.cuda(local_rank)
#                 pet_images = pet_images.cuda(local_rank)
#                 bboxes = [ann.cuda(local_rank) for ann in bboxes]
#                 # bboxes = bboxes.cuda(local_rank)
#         # targets = [ann.cuda(local_rank) for ann in targets]
#         #----------------------#
#         #   清零梯度
#         #----------------------#
#         optimizer.zero_grad()
#         # print(bboxes)
#         # first_image_np = ct_images[0].cpu().numpy().transpose(1, 2, 0)
        
#         # 4. 如果图像是归一化的（例如，值在-1到1之间或0到1之间），需要反归一化
#         # 假设图像归一化到 [0, 1] 范围
#         # if first_image_np.min() < 0 or first_image_np.max() > 1:
#         #     # 如果值不在[0,1]范围内，进行归一化以便显示
#         #     first_image_np = (first_image_np - first_image_np.min()) / (first_image_np.max() - first_image_np.min())
        
#         # # 5. 绘制图像
#         # plt.figure(figsize=(10, 10))
#         # plt.imshow(first_image_np)
#         # plt.axis('off')  # 关闭坐标轴
#         # plt.title('First Image from Batch')
#         # plt.show()
        
#         # 如果您想要保存图像
#         # plt.savefig('/root/autodl-tmp/detection/model/yolox-pytorch-main/nets/first_image.png', bbox_inches='tight', pad_inches=0)

        
#         if not fp16:
#             #----------------------#
#             #   前向传播 - 两个模态分别输入
#             #----------------------#
#             outputs = model_train(ct_images, pet_images)
#             loss_value = yolo_loss(outputs, bboxes)
            
#             #----------------------#
#             #   反向传播
#             #----------------------#
#             loss_value.backward()
#             torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
#             optimizer.step()
#         else:
#             from torch.cuda.amp import autocast
#             with autocast():
#                 #----------------------#
#                 #   前向传播 - 两个模态分别输入
#                 #----------------------#
#                 outputs = model_train(ct_images, pet_images)
#                 loss_value = yolo_loss(outputs, bboxes)

#             #----------------------#
#             #   反向传播
#             #----------------------#
#             scaler.scale(loss_value).backward()
#             scaler.unscale_(optimizer)  # unscale gradients
#             torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
#             scaler.step(optimizer)
#             scaler.update()
        
#         if ema:
#             ema.update(model_train)

#         loss += loss_value.item()
        
#         if local_rank == 0:
#             pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
#                                 'lr'    : get_lr(optimizer)})
#             pbar.update(1)

#     if local_rank == 0:
#         pbar.close()
#         print('Finish Train')
#         print('Start Validation')
#         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

#     if ema:
#         model_train_eval = ema.ema
#     else:
#         model_train_eval = model_train.eval()
        
#     for iteration, batch in enumerate(gen_val):
#         if iteration >= epoch_step_val:
#             break
        
#         # 多模态：batch[0] 是一个元组 (ct_images, pet_images), batch[1] 是 bboxes
#         images, bboxes = batch
#         ct_images, pet_images = images  # 解包得到CT和PET图像
        
#         with torch.no_grad():
#             if cuda:
#                 ct_images = ct_images.cuda(local_rank)
#                 pet_images = pet_images.cuda(local_rank)
#                 # bboxes = bboxes.cuda(local_rank)
#                 bboxes = [ann.cuda(local_rank) for ann in bboxes]
            
#             #----------------------#
#             #   前向传播 - 两个模态分别输入
#             #----------------------#
#             outputs = model_train_eval(ct_images, pet_images)
#             loss_value = yolo_loss(outputs, bboxes)

#         val_loss += loss_value.item()
#         if local_rank == 0:
#             pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
#             pbar.update(1)
            
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Validation')
#         loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
#         eval_callback.on_epoch_end(epoch + 1, model_train_eval)
#         print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
#         print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
#         #-----------------------------------------------#
#         #   保存权值
#         #-----------------------------------------------#
#         if ema:
#             save_state_dict = ema.ema.state_dict()
#         else:
#             save_state_dict = model.state_dict()

#         if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
#             torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
#         if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
#         torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
def fit_one_epoch2(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer,
                   epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler,
                   save_period, save_dir, local_rank=0):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, bboxes = batch[0], batch[1]
        ct_images, pet_images = images
        with torch.no_grad():
            if cuda:
                ct_images = ct_images.cuda(local_rank)
                pet_images = pet_images.cuda(local_rank)
                bboxes = [ann.cuda(local_rank) for ann in bboxes]

        optimizer.zero_grad()

        if not fp16:
            outputs = model_train(ct_images, pet_images)
            loss_value = yolo_loss(outputs, bboxes)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(ct_images, pet_images)
                loss_value = yolo_loss(outputs, bboxes)
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch
        ct_images, pet_images = images
        with torch.no_grad():
            if cuda:
                ct_images = ct_images.cuda(local_rank)
                pet_images = pet_images.cuda(local_rank)
                bboxes = [ann.cuda(local_rank) for ann in bboxes]
            outputs = model_train_eval(ct_images, pet_images)
            loss_value = yolo_loss(outputs, bboxes)
        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)

        # ---------- 1. 执行回调，计算验证集 mAP ----------
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)

        # ---------- 2. 判断本轮是否计算了新的 mAP ----------
        current_map = None
        if hasattr(eval_callback, 'epoches') and len(eval_callback.epoches) > 0:
            # 如果最后一个记录的 epoch 等于当前 epoch，说明本轮计算了 mAP
            if eval_callback.epoches[-1] == epoch + 1:
                current_map = eval_callback.maps[-1]   # 获取本轮 mAP

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f' % (loss / epoch_step, val_loss / epoch_step_val))
        if current_map is not None:
            print('Current mAP: %.4f' % current_map)

        # ---------- 3. 准备保存的权值 ----------
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        # 周期保存（按 save_period）
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir,
                       "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        # ---------- 4. 基于验证集 mAP 保存最佳模型 ----------
        # 使用函数属性记录历史最佳 mAP
        if not hasattr(fit_one_epoch2, "best_map"):
            fit_one_epoch2.best_map = 0.0

        if current_map is not None and current_map > fit_one_epoch2.best_map:
            print('Save best model (mAP = %.4f) to best_epoch_weights.pth' % current_map)
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            fit_one_epoch2.best_map = current_map

        # 始终保存最新模型
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))