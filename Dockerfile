FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y openssh-server
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_con>
RUN ssh-keygen -A
RUN mkdir -p /run/sshd
RUN echo 'root:0000' | chpasswd

CMD ["/usr/sbin/sshd", "-D"]
