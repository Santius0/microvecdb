#!/bin/bash

mkdir jetson-nano-sysroot
#rsync -avz jetson-nano-ip:/lib ~/jetson-nano-sysroot/
#rsync -avz jetson-nano-ip:/usr/lib ~/jetson-nano-sysroot/usr/
#rsync -avz jetson-nano-ip:/usr/include ~/jetson-nano-sysroot/usr/

rsync -avzHAX santius@192.168.55.1:/lib /home/santius/microvecdb/jetson/jetson-nano-sysroot
rsync -avzHAX santius@192.168.55.1:/usr/lib /home/santius/microvecdb/jetson/jetson-nano-sysroot/usr
rsync -avzHAX santius@192.168.55.1:/usr/include /home/santius/microvecdb/jetson/jetson-nano-sysroot/usr

#sudo rsync -aAXv --exclude={"/dev/*","/proc/*","/sys/*","/tmp/*","/run/*","/mnt/*","/media/*","/lost+found"} santius@192.168.55.1:/ /home/santius/microvecdb/microvecdb/jetson/jetson-nano-sysroot/