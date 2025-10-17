import subprocess
import psutil

def get_gpu_memory():
    try:
        # Run the nvidia-smi command
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            # Print the output
            total, used, free = result.stdout.strip().split(',') 
            # print(f"Total Memory: {total} MB")
            # print(f"Used Memory: {used} MB")
            # print(f"Free Memory: {free} MB")
            return 100*(float(used)/float(total))
            # print(f"VRAM Usage Percentage: {100*(float(used)/float(total)):.1f}%")
        else:
            print("Error retrieving GPU memory information:", result.stderr)
    except Exception as e:
        print("An error occurred:", str(e))

def get_ram_memory():

    # Get the virtual memory details
    memory_info = psutil.virtual_memory()

    # Print memory information
    # print(f"Total RAM: {memory_info.total / (1024 ** 2):.2f} MB")
    # print(f"Used RAM: {memory_info.used / (1024 ** 2):.2f} MB")
    # print(f"Free RAM: {memory_info.free / (1024 ** 2):.2f} MB")
    # print(f"Available RAM: {memory_info.available / (1024 ** 2):.2f} MB")
    return memory_info.percent
    # print(f"RAM Usage Percentage: {memory_info.percent}%")

# get_ram_memory()
# get_gpu_memory()