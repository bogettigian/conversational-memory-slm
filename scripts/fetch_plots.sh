#!/bin/bash

rsync -avz -e "ssh -i aws.pem" ubuntu@18.188.119.33:/home/ubuntu/investigaton/plots ./
