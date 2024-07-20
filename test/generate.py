import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from generator.generator import main

if __name__ == "__main__":
    main(n=5)