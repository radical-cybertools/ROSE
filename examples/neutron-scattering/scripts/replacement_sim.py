#!/usr/bin/env python
import time
import sys

def main():
    print("Arguments:", sys.argv)
    print("Sleeping for 10 seconds")
    time.sleep(10)
    print("Waking up")
    
if __name__ == '__main__':
    main()

