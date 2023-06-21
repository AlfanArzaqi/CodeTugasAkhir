import socket
import time
from pynput import keyboard

my_socket = socket.socket()

port = 5055

ip = "192.168.1.6"

my_socket.connect((ip, port))
msg = str(0)

def on_key_realese(key):
    global result
    if result != 'stop' :
        result = 'stop'
        my_socket.send(result.encode('utf_8'))

def on_key_pressed(key) :
    

while True :
    # msg = (my_socket.recv(1024).decode())
    # print(msg,end='')

    msg = input()
    my_socket.send(msg.encode('utf_8'))