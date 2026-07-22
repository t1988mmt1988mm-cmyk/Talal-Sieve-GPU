import torch
import time
import math

def talal_sieve_torch(n):
    device = torch.device("cuda")
    sieve = torch.ones(n+1, dtype=torch.bool, device=device)
    sieve[:2] = False
    limit = int(math.sqrt(n)) + 1
    for i in range(2, limit):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    primes = torch.where(sieve)[0]
    return primes.cpu()

if __name__ == "__main__":
    N = 10**9
    torch.cuda.synchronize()
    start = time.time()
    primes = talal_sieve_torch(N)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time: {end-start:.3f} seconds")
    print(f"Primes: {len(primes):,}")
