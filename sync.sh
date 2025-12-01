rsync -avz -e "ssh -i aws.pem" --filter=':- .gitignore' --exclude 'sync.sh' --exclude '.git' . ubuntu@18.188.119.33:~/investigaton
