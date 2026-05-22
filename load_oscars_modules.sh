#!/bin/bash
#clear

module_loaded() {
    module list 2>&1 | grep -q "$1"
}

module_loaded cmake/3.29.6-ocf3 || module load cmake/3.29.6-ocf3
module_loaded opencv/4.6.0s-22w7 || module load opencv/4.6.0s-22w7
module_loaded cuda/12.9.0-cinr || module load cuda/12.9.0-cinr