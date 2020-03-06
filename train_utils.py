import os, torch

def save_checkpoint(epoch, model, optimizer, optimizer_sparse=None, scheduler=None, scheduler_sparse=None, suffix="default"):
    if optimizer_sparse is None:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, "checkpoint-{}.pth".format(suffix))
    else:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_sparse': optimizer_sparse.state_dict(),
        }, "checkpoint-{}.pth".format(suffix))

def from_checkpoint_if_exists(model, optimizer, optimizer_sparse=None, scheduler=None, scheduler_sparse=None):
    epoch = 0
    if os.path.isfile("checkpoint.pth"):
        print("Loading existing checkpoint...")
        checkpoint = torch.load("checkpoint.pth")
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if optimizer_sparse is not None:
            optimizer_sparse.load_state_dict(checkpoint['optimizer_sparse'])
            scheduler_sparse.load_state_dict(checkpoint['scheduler_sparse'])
            

            return epoch, model, optimizer, optimizer_sparse, scheduler, scheduler_sparse

            # optimizer_sparse.load_state_dict(checkpoint['optimizer_sparse'])
    if optimizer_sparse is not None:
        return epoch, model, optimizer, optimizer_sparse, scheduler, scheduler_sparse
    
    return epoch, model, optimizer, scheduler

def tb_mle_meta_batch(tb, loss, acc, num_iter):
    tb.add_scalars(
        {
            "loss_meta_batch": loss,
            "accuracy_meta_batch": acc,
        },
        group="meta_mle_train",
        sub_group="batch",
        global_step=num_iter
    )


def tb_mle_epoch(tb, loss_per_word, accuracy, epoch):
    tb.add_scalars(
        {
            "loss_per_word" : loss_per_word,
            "accuracy" : accuracy,
        },
        group="train",
        sub_group="epoch",
        global_step=epoch
    )

def tb_bleu_validation_epoch(tb, bleu, accuracy, epoch):
    tb.add_scalars(
        {
            "validation_bleu" : bleu,
            "accuracy" : accuracy,
        },
        group="train",
        sub_group="epoch",
        global_step=epoch
    )

def tb_mle_batch(tb, total_loss, n_word_total, n_word_correct, epoch, batch_idx, data_len):
    tb.add_scalars(
        {
            "loss_per_word" : total_loss / n_word_total,
            "accuracy": n_word_correct / n_word_total,
        },
        group="mle_train",
        sub_group="batch",
        global_step = epoch*data_len+batch_idx
    )