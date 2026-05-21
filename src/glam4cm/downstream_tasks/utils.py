
def get_logging_steps(dataset_size, num_epochs, batch_size):
    """
    Calculate the logging steps based on the dataset size, number of epochs, and batch size.
    """
    num_steps = dataset_size // batch_size
    logging_steps = num_steps * num_epochs // 20
    print(f"Logging steps: {logging_steps}")
    return logging_steps