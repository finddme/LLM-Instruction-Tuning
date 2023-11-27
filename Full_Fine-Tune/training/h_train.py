def train(huggingface_ck,
          batch_size,
          train_dataset,
         gradient_accumulation_steps,
         optimizer=torch.optim.Optimizer,
         max_grad_norm,
         lr_scheduler=torch.optim.lr_scheduler.LambdaLR):
    
    steps_skipped = 0
    epoch= 0
    for step, inputs in enumerate(train_dataset):
        #training_step
        model.train()
        outputs = model(inputs)
        loss = outputs["loss"]
        loss = loss.mean()
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(loss).backward()
        tr_loss_step  = loss.detach() / gradient_accumulation_steps
        tr_loss += tr_loss_step
        # Gradient clipping
        scaler.unscale_(optimizer)
        model.clip_grad_norm_(max_grad_norm)
        # Optimizer step
        optimizer_was_run = True
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        optimizer_was_run = scale_before <= scale_after
        
        model.zero_grad()
        global_step += 1
        epoch += 1