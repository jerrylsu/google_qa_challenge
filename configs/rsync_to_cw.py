#!/usr/bin/python3
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rsync server")

    parser.add_argument(
        "rsync",
        choices=["push", "pull"],
        help="Rsync with server")
        
    args = parser.parse_args()
    
    if args.rsync == 'push':
        print('Starting to pull...')
        pull_cmd = 'rsync -avh --progress \
                    $c:/home/yckj2939/project/google_qa_challenge/ \
                    /home/YCKJ2939/project/jerry/kaggle/google_qa_challenge/'
        os.system(pull_cmd)
        print('Starting to push...')
        push_cmd = 'rsync -avh --progress --exclude=.git --exclude=.idea\
                    /home/YCKJ2939/project/jerry/kaggle/google_qa_challenge/ \
                    $c:/home/yckj2939/project/google_qa_challenge/'
        os.system(push_cmd)
    elif args.rsync == 'pull':
        print('Starting to pull...')
        pull_cmd = 'rsync -avh --progress \
                   $c:/home/yckj2939/project/google_qa_challenge/ \
                   /home/YCKJ2939/project/jerry/kaggle/google_qa_challenge/'
        os.system(pull_cmd)

