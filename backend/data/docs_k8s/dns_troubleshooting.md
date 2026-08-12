# DNS 解析故障排查

## 集群内服务名解析失败

### 现象
Pod 内执行 `nslookup <service-name>` 无法解析出对应的 Service IP，报
`can't resolve` 一类错误，应用连接内部服务时报 DNS 解析失败或超时。

### 根因
集群内的服务名解析依赖 CoreDNS（或其前身 kube-dns）。任何一个环节出问题都
会导致解析失败：CoreDNS Pod 本身未运行、DNS Service 没有正常暴露端点、
CoreDNS 缺少读取 Service/EndpointSlice 的 RBAC 权限、Pod 的 `resolv.conf`
配置异常，或者查询没有带上正确的命名空间——DNS 查询若不显式指定命名空间，
默认只在发起查询的 Pod 自身所在命名空间内解析，跨命名空间访问 Service 时
必须显式带上目标命名空间（`<service-name>.<namespace>`），否则会被当作
"这个名字在当前命名空间不存在"处理。

### 处理步骤
1. 检查 Pod 内 `/etc/resolv.conf` 是否配置了合理的 `search` 和 `nameserver`
   （`kubectl exec -ti <pod> -- cat /etc/resolv.conf`）
2. 确认 CoreDNS Pod 处于 `Running` 状态：
   `kubectl get pods --namespace=kube-system -l k8s-app=kube-dns`
3. 确认 DNS Service 本身存在且有可用端点：
   `kubectl get svc --namespace=kube-system` 和
   `kubectl get endpointslice -l kubernetes.io/service-name=kube-dns --namespace=kube-system`
4. 查看 CoreDNS 日志排查是否有异常（`kubectl logs --namespace=kube-system -l k8s-app=kube-dns`），
   必要时给 CoreDNS 的 Corefile 临时加上 `log` 插件，确认查询确实被接收到
5. 若报错是 `SERVFAIL`，检查 `system:coredns` 这个 ClusterRole 是否具备
   `services`/`endpoints`/`endpointslices` 等资源的 `list`/`watch` 权限——
   权限缺失会导致 CoreDNS 无法正确解析服务名，但这类根因很容易被误判为
   "网络问题"而不是"权限问题"
6. 确认查询是否跨命名空间：若目标 Service 与发起查询的 Pod 不在同一命名空间，
   查询必须写成 `<service-name>.<namespace>` 而不是只写服务名
7. 部分 Linux 发行版（如 Ubuntu）默认用 `systemd-resolved` 接管本地 DNS，
   会替换 `/etc/resolv.conf` 为一个存根文件，可能导致解析环路——这类问题
   需要在 kubelet 层通过 `--resolv-conf` 指向正确的解析配置文件解决，
   不是 Kubernetes 应用层能直接修复的

---
来源：[Kubernetes 官方文档 - 调试 DNS 问题](https://kubernetes.io/zh-cn/docs/tasks/administer-cluster/dns-debugging-resolution/)，
CC BY 4.0 授权，节选整理。
