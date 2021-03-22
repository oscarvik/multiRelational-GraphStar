import torch

def get_memory_info():
    if(torch.cuda.is_available())
        torch.cuda.memory_summary(device=None, abbreviated=False)
        gpu_count = torch.cuda.device_count()
        print(f"GPU count: {gpu_count}")
        for num in range(0,gpu_count):
            t = torch.cuda.get_device_properties(num).total_memory
            r = torch.cuda.memory_reserved(num) 
            a = torch.cuda.memory_allocated(num)
            f = r-a  # free inside reserved
            print(f"""
            GPU number: {num}
            Total: {t}, 
            Reserved: {r}, 
            Allocated: {a}, 
            Free: {f}""")
    else:
        print("Using CPU.")