"""
paging_stress_test.py - KV-Cache Paging & OOM Compaction Benchmark
"""

import sys
import time
sys.path.insert(0, ".")
from ore_client import OreClient

def main():
    ore = OreClient()
    
    print("==================================================")
    print("  ORE KERNEL :: KV-CACHE PAGING STRESS TEST")
    print("==================================================\n")

    # 1. WIPE THE SLATE CLEAN
    print("[*] Wiping previous memory state...")
    ore.clear("openclaw")
    print("[+] Memory wiped.\n")

    # 2. COLD START
    print("[*] TEST 1: Cold Start & Page Out")
    print("    Prompt: 'My secret code is 8842 and my favorite food is pizza.'")
    
    start_time = time.time()
    res1 = ore.ask("My secret code is 8842 and my favorite food is pizza. Just reply 'Got it'.")
    end_time = time.time()
    
    print(f"    AI: {res1.strip()}")
    print(f"    Latency (Cold Start): {end_time - start_time:.2f} seconds\n")

    # 3. PAGE IN (ZERO-LATENCY)
    print("[*] TEST 2: Page In (Fast-Forward)")
    print("    Prompt: 'What is my secret code and favorite food?'")
    
    start_time = time.time()
    res2 = ore.ask("What is my secret code and favorite food?")
    end_time = time.time()
    
    print(f"    AI: {res2.strip()}")
    print(f"    Latency (KV-Cache Page In): {end_time - start_time:.2f} seconds\n")

    # 4. HARDWARE EVICTION SURVIVAL
    print("[*] TEST 3: Hardware Eviction Survival")
    print("    Simulating a heavily loaded GPU by force-evicting the model...")
    # NOTE: Change this if your manifest uses a different model!
    ore.expel("llama3.2:1b") 
    
    print("    Prompt: 'Are you sure about the pizza?'")
    start_time = time.time()
    res3 = ore.ask("Are you sure about the pizza?")
    end_time = time.time()
    
    print(f"    AI: {res3.strip()}")
    print(f"    Latency (Model Boot + KV Inject): {end_time - start_time:.2f} seconds\n")

    print("    Prompt: 'what is 2 * 4?'")
    start_time = time.time()
    res4 = ore.ask("what is 2 * 4?")
    end_time = time.time()
    print(f"    AI: {res4.strip()}")
    print(f"    Latency KV Inject: {end_time - start_time:.2f} seconds\n")

    print("    Prompt: 'what is my secret code?'")
    start_time = time.time()
    res5 = ore.ask("what is my secret code?")
    end_time = time.time()
    print(f"    AI: {res5.strip()}")
    print(f"    Latency KV Inject: {end_time - start_time:.2f} seconds\n")

    # 5. OOM COMPACTION ENGINE
    print("[*] TEST 4: The Compaction Engine (OOM Defense)")
    print("    Flooding the agent with contextual history to breach the 200-token limit...")
    
    # A massive block of text hiding the secret code inside it.
    flood_text = (
        "I want to tell you about the history of Operating Systems and Memory Management. "
        "In the early days of computing, physical memory was extremely limited, requiring developers "
        "to manually load and unload segments of their programs. This evolved into Virtual Memory, "
        "where the OS abstracts physical RAM into virtual address spaces. Paging allows the OS to "
        "divide memory into fixed-size blocks (pages) and move inactive pages to a slower secondary "
        "storage like an SSD (a swap file) to free up physical RAM. When a program needs that memory "
        "again, the CPU triggers a Page Fault, and the OS fetches the data back from disk. "
        "Modern hypervisors use similar concepts for managing virtual machines. "
        "Rust adds another layer of complexity by enforcing strict ownership rules at compile-time, "
        "meaning the memory is managed without a garbage collector. "
        "By the way, amidst all this technical discussion, do not forget my secret code is 8842 and I still love pizza. "
        "Anyway, back to systems architecture. To achieve low latency context switching, "
        "an OS must minimize the time it takes to move state between disk and compute. "
        "This is similar to how AI models manage KV-Caches, storing the attention matrices so they "
        "don't have to recompute past tokens. Please acknowledge you received this history lesson."
    )

    start_time = time.time()
    res6 = ore.ask(flood_text)
    end_time = time.time()
    
    print(f"    AI: {res6.strip()}")
    print(f"    Latency (Processing Flood): {end_time - start_time:.2f} seconds")
    print("    [!] Flood complete. Check the ORE Server Terminal for the [COMPACTION] logs!\n")

    # 6. POST-COMPACTION RECALL
    # print("[*] TEST 4: The 'Rebirth' ( Recall)")
    # print("    Waiting 5 seconds for the background compaction thread to finish summarizing...\n")
    # time.sleep(5) 
    
    print("    Prompt: 'What is my favorite food again?'")
    start_time = time.time()
    res7 = ore.ask("what is my favorite food?")
    end_time = time.time()
    
    print(f"    AI: {res7.strip()}")
    print(f"    Latency (Post-Compaction Cold Start): {end_time - start_time:.2f} seconds\n")

    print("==================================================")
    print("  STRESS TEST COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()