---
layout: article
title: Samba服务端与客户端安装配置
date: 2019-02-21 00:07:00 +0800
tag: [Tutorials, Linux] 
categories: Tutorials
pageview: true
---


# Samba 文件共享配置

## 一、CentOS服务端---Win客户端

### 1.1 安装samba组件

```
yum install samba samba-client samba-swat
```

- 查看安装情况：

![k2I95q.png](https://s2.ax1x.com/2019/02/20/k2I95q.png)

### 1.2 配置samba

- 设置开机启动：

```
chkconfig smb on
chkconfig nmb on
```

- 新建访问用户

```
useradd xxx       # 新建用户
smbpasswd -a xxx  # 修改密码   
```

- 创建共享文件夹

```
mkdir /root/root_share
# 修改共享文件夹权限
cd /
chmod 777 /root
chmod 777 /root/root_share
```

- 修改samba配置文件

```
vi /etc/samba/smb.conf
```

![k2ItZd.png](https://s2.ax1x.com/2019/02/20/k2ItZd.png)

- 关闭防火墙

```
chkconfig iptables off
```

- 关闭SELINUX

```
vim /etc/selinux/config
```

![k2IUII.png](https://s2.ax1x.com/2019/02/20/k2IUII.png)



- 系统重启

```
reboot
```

- 查看samba启动状态

```
service smb status
```

![k2IOF1.png](https://s2.ax1x.com/2019/02/20/k2IOF1.png)



- 修改host allow

```
vim /etc/samba/smb.conf
```

![k2oCeH.png](https://s2.ax1x.com/2019/02/20/k2oCeH.png)

> 注意不修改的话，windows无法访问



### 1.3 客户端连接共享文件夹

> 客户端环境：windows10家庭版

- 开启SMB/CIFS支持

![k2o1kn.png](https://s2.ax1x.com/2019/02/20/k2o1kn.png)

- 本地组策略编辑

按住快捷键Win+R打开运行窗口，往运行里面输入**gpedit.msc** 打开的是组策略编辑器

![k2oBkR.png](https://s2.ax1x.com/2019/02/20/k2oBkR.png)



> 注意：有些用户想要打开组策略编辑器却遇到了gpedit.msc找不到的提示
>
> ![k2oyp6.png](https://s2.ax1x.com/2019/02/20/k2oyp6.png)
>
> 解决方法：https://pan.baidu.com/s/1s9Il6ifEvXzGEUAiZ65GHg
>
> 下载上述文件，右键单击这个“win10添加策略组.cmd”文件，选择以**管理员身份运行**即可，运行完毕，系统成功加入策略组。



- 重启电脑



- windows访问共享目录

  ![k2TZNR.png](https://s2.ax1x.com/2019/02/20/k2TZNR.png)

  ![k2TVE9.png](https://s2.ax1x.com/2019/02/20/k2TVE9.png)

  



- 访问成功





## 二、 -- Ubuntu客户端

### 2.1 安装samba客户端组件

```
sudo apt-get install smbclient
```

###  2.2 查看所以共享目录

```
smbclient -L server_ip
```

> 注：敲入上面命令后，在出现提示输入密码时，直接按Enter键（因为此处是匿名访问），结果会显示指定Samba服务器上当前全部的共享目录。



### 2.3 连接共享目录

```
smbclient //server_ip/root/root_share
```



### 2.4 挂载

```
mount -t cifs -o username=xxx,password=xxx //server_ip/root/root_share 本地挂载点
```



### 2.5 关于权限

