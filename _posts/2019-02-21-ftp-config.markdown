---
layout: article
title: FTP服务端与客户端安装配置
date: 2019-02-21 00:07:00 +0800
tag: [Config, Linux] 
categories: Tutorials
pageview: true
---


# FTP文件传输配置

## 一、CentOS服务器端

### 1. 安装vsftpd

```
yum install vsftpd
```

### 2.设置开机启动vsftpd ftp服务

```
chkconfig vsftpd on
```

### 3.启动vsftpd服务

```
service vsftpd start
```

### 4.配置防火墙

因为ftp默认的端口为21，而centos默认是没有开启的，所以要修改iptables文件

```
vi /etc/sysconfig/iptables
```

在行上面有`22 -j ACCEPT` 下面另起一行输入 **这行代码**：

```
-A RH-Firewall-1-INPUT -m state --state NEW -m tcp -p tcp --dport 21 -j ACCEPT
```

保存关闭，重启防火墙。

> 我已经永久关闭防火墙了，所以也无需开启

### 5.配置vsftpd服务器

打开配置文件

```
vi /etc/vsftpd/vsftpd.conf
```

- 把第一行的 `anonymous_enable=YES` ，改为`NO`，取消匿名登陆

  > 也可不取消

- ```
  #chroot_list_enable=YES
  # (default follows)
  #chroot_list_file=/etc/vsftpd.chroot_list
  ```

  改为

  ```
  chroot_list_enable=YES
  # (default follows)
  chroot_list_file=/etc/vsftpd/chroot_list
  ```

- 修改权限，使用户能重命名或删除

  ```
  anon_mkdir_write_enable=YES
  anon_other_write_enable=YES   
  ```

- 设置登陆默认路径（共享文件夹）

  ```
  local_root=/share
  ```

- 其他设置

```
write_enables=YES
```

保存关闭，重启 `vsftpd` 服务



### 6. 添加ftp用户

添加一个名为 ftpuser 的用户，所属 ftp 用户组，指向目录/share, 禁止登录：

```
useradd -d /share -g ftp -s /sbin/nologin ftpuser
```

设置ftpuser登陆密码：

```
passwd ftpuser
```

添加FTP用户到 `user_list`文件夹中：

```
# 打开user_list文件
vi /etc/vsftpd/user_list

# 文件内文末添加
ftpuser
```

添加FTP用户到 `chroot_list` 文件夹：

```
# 在/etc/vsftpd/ 目录下创建一个 chroot_list 文件：
vi /etc/vsftpd/chroot_list

# 文件内文末添加
ftpuser
```

> 或者可以利用以有的用户

### 7.修改selinux

```
setsebool -P allow_ftpd_full_access 1   

setsebool -P ftp_home_dir off 1 
```

重启vsftpd







## 二. Windows10客户端

### 2.1 开启windows相关功能

![kfkc7Q.png](https://s2.ax1x.com/2019/02/22/kfkc7Q.png)

- 开启后重启计算机



### 2.2 快速连接

- 在文件夹路径窗口输入 `ftp://10.5.18.250`，连接

![kfA0UJ.png](https://s2.ax1x.com/2019/02/22/kfA0UJ.png)



连接成功：

![kfA6v6.png](https://s2.ax1x.com/2019/02/22/kfA6v6.png)



> 到此结束也ok，若想要建立长久的本地驱动映射，继续



### 2.3 建立本地驱动映射

- 打开 `此电脑-->计算机-->映射网络驱动器` :

![kfA9BD.png](https://s2.ax1x.com/2019/02/22/kfA9BD.png)



- 选择 `连接到可用于存储文档和图片的网站 `

![kfk5cV.png](https://s2.ax1x.com/2019/02/22/kfk5cV.png)





- 

![kfEGIH.png](https://s2.ax1x.com/2019/02/22/kfEGIH.png)



- ![kfEfyV.png](https://s2.ax1x.com/2019/02/22/kfEfyV.png)
- 最后输入密码