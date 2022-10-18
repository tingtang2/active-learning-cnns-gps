import argparse
import sys
from src.models.base_cnn import BaseCNN

def main() -> int:
    model = BaseCNN()
    return 0

if __name__ == '__main__':
    sys.exit(main())