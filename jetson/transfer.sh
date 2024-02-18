#!/bin/bash

source_file=""
destination_dir=""

print_usage() {
    echo "Usage: $0 [-s source_file] [-d destination_dir]"
    echo "Options:"
    echo "  -s    Specify the source file to transfer."
    echo "  -d    Specify the destination directory."
    echo "If any option is not provided, the script will prompt for it."
}

while getopts "s:d:" opt; do
  case ${opt} in
    s )
      source_file=$OPTARG
      ;;
    d )
      destination_dir=$OPTARG
      ;;
    \? )
      print_usage
      exit 1
      ;;
  esac
done

if [ -z "$source_file" ]; then
    read -p "Enter the source file to transfer: " source_file
fi

if [ -z "$destination_dir" ]; then
    read -p "Enter the destination directory: " destination_dir
fi

password_file="jetson_password.txt"

if ! command -v sshpass &> /dev/null
then
    echo "sshpass could not be found. Please install it to use this script."
    exit 1
fi

if [ ! -f "$password_file" ]; then
    echo "Password file does not exist. Please create jetson_password.txt with the password."
    exit 1
fi

password=$(cat "$password_file")

echo "$source_file ====> santius@192.168.55.1:$destination_dir/$source_file"

sshpass -p "$password" scp "$source_file" "santius@192.168.55.1:$destination_dir"

echo "File transfer complete."
