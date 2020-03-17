   
for batch_idx, (subject) in enumerate(test_loader):
    with torch.no_grad():
        image_1 = subject['image_1']
        image_2 = subject['image_2']
        image_3 = subject['image_3']
        image_4 = subject['image_4']

        mask = subject['gt']
        b,c,x,y,z = mask.shape
        image_1, image_2, image_3, image_4, mask = image_1.cuda(),image_2.cuda(),image_3.cuda(),image_4.cuda(), mask.cuda()

        output_1 = model(image_1.float())
        output_2 = model(image_2.float())
        output_3 = model(image_3.float())
        output_4 = model(image_4.float())

        output = torch.tensor(np.zeros((b,c,x,y,z)))

        output[:,:,0:144,0:144,:] = output_1
        output[:,:,x-144:x,0:144,:] = output_2
        output[:,:,0:144,y-144:y,:] = output_3
        output[:,:,x-144:x,y-144:y,:] = output_4
        output = output.cuda()

        curr_loss = dice_loss(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()).cpu().data.item()
        total_loss+=curr_loss
        # Computing the average loss
        average_loss = total_loss/(batch_idx + 1)
        #Computing the dice score 
        curr_dice = 1 - curr_loss
        #Computing the total dice
        total_dice+= curr_dice
        #Computing the average dice
        average_dice = total_dice/(batch_idx + 1)
        print("Current Dice is: ", curr_dice)


print("Average dice is: ", average_dice)
