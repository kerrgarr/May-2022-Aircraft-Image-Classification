import torch
import torch.utils.tensorboard as tb
import os

def tensorboard_logging(train_logger, valid_logger):
    """
    Tensorboard logging of accuracy and loss
    """

    global_step = 0
    for epoch in range(10):
        torch.manual_seed(epoch)
        total_accuracy = []
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            train_logger.add_scalar('loss', dummy_train_loss, global_step=global_step)
            total_accuracy.append(dummy_train_accuracy)
            global_step += 1
        train_logger.add_scalar('accuracy', torch.mean(torch.cat(total_accuracy)), global_step=global_step)

        total_accuracy = []
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            total_accuracy.append(dummy_validation_accuracy)
        valid_logger.add_scalar('accuracy', torch.mean(torch.cat(total_accuracy)), global_step=global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    print(os.path.join(args.log_dir, 'train'))
    train_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'test'))
    tensorboard_logging(train_logger, valid_logger)
