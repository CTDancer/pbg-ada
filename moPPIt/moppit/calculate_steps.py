import math


# def calculate_steps_per_epoch(total_samples, batch_size_per_gpu, num_gpus, scheduling):
#     # Calculate total batch size across all GPUs
#     total_batch_size = batch_size_per_gpu * num_gpus
#
#     # Calculate total batches per epoch
#     batches_per_epoch = math.ceil(total_samples / total_batch_size)
#
#     steps_per_epoch = []
#     current_accumulation_factor = 1  # Default accumulation factor
#
#     for epoch in range(max(scheduling.keys()) + 1):
#         # Update accumulation factor if it's defined for the current epoch
#         if epoch in scheduling:
#             current_accumulation_factor = scheduling[epoch]
#
#         effective_steps = math.ceil(batches_per_epoch / current_accumulation_factor)
#         steps_per_epoch.append(effective_steps)
#
#     return steps_per_epoch

def calculate_total_steps(total_samples, batch_size, num_gpus, accumulation_schedule, max_epochs):
    total_steps = 0

    for epoch in range(max_epochs):
        # Determine the accumulation steps for the current epoch
        for start_epoch, steps in accumulation_schedule.items():
            if start_epoch > epoch:
                break
            accumulation_steps = steps

        effective_batch_size = batch_size * num_gpus * accumulation_steps
        steps_per_epoch = (total_samples + effective_batch_size - 1) // effective_batch_size

        total_steps += steps_per_epoch
        print(f'Epoch {epoch}: {steps_per_epoch} steps (accumulation_steps={accumulation_steps})')

    return total_steps


# total_samples = 12284  # Replace with the actual number of samples in your dataset
# batch_size = 32
# num_gpus = 6
# accumulation_schedule = {0:4, 5:2, 30:1}
# max_epochs = 50
#
# total_steps = calculate_total_steps(total_samples, batch_size, num_gpus, accumulation_schedule, max_epochs)
# print(f"Total Steps: {total_steps}")

total_samples = 306482  # Replace with the actual number of samples in your dataset
batch_size = 32
num_gpus = 6
accumulation_schedule = {0:8, 3:4, 20:2}
max_epochs = 30

total_steps = calculate_total_steps(total_samples, batch_size, num_gpus, accumulation_schedule, max_epochs)
print(f"Total Steps: {total_steps}")

#
# # Example usage
# total_samples = 309503
# batch_size_per_gpu = 16
# num_gpus = 7
# scheduling = {0: 4, 5: 3, 10: 2, 13: 1}
#
# steps_per_epoch = calculate_steps_per_epoch(total_samples, batch_size_per_gpu, num_gpus, scheduling)
# for epoch, steps in enumerate(steps_per_epoch):
#     print(f"Epoch {epoch}: {steps} steps")
#
# print(f"Total steps: {sum(steps_per_epoch)}")
