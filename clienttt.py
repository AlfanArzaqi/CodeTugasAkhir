import socket

my_socket = socket.socket()

port = 8876

ip = "192.168.5.51"

my_socket.connect((ip, port))
msg = str(0)

while True :
    msg = (my_socket.recv(1024).decode())
    print(msg,end='')

    # msg = input()
    # my_socket.send(msg.encode('utf_8'))