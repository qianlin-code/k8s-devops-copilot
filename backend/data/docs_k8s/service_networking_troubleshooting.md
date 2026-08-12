# Service 与网络访问故障排查

## Service 无法访问（Endpoints 为空）

### 现象
通过 Deployment 运行了 Pod 并创建了对应 Service，但访问 Service 时没有任何
响应，或者一直连接超时。

### 根因
最常见原因是 Service 的 `selector` 与 Pod 的标签（labels）不匹配，导致
Service 没有关联到任何 Pod，对应的 EndpointSlice 为空。此外，Pod 的
`containerPort` 与 Service 的 `targetPort` 不一致也会导致流量转发失败，
即使 Endpoints 不为空、连接也会被拒绝或超时。

### 处理步骤
1. 执行 `kubectl get endpointslices -l kubernetes.io/service-name=<SERVICE_NAME>`
   查看该 Service 关联的端点数量
2. 若数量为 0 或明显少于预期的 Pod 副本数：用 Service 的 `selector` 字段去
   反查 Pod，执行 `kubectl get pods --selector=<与 Service selector 相同的键值>`，
   核对返回的 Pod 列表是否是期望作为该 Service 后端的那批 Pod
3. 若 selector 匹配没问题，检查 Pod 定义里的 `containerPort` 是否与 Service
   的 `targetPort` 一致——两者必须对应同一个容器监听端口
4. 确认应用本身确实在容器内监听了该端口（可用调试容器执行
   `kubectl exec` 进入 Pod 网络命名空间验证）

## Node 状态显示 NotReady

### 现象
`kubectl get nodes` 输出中某个节点的 `STATUS` 显示为 `NotReady`，该节点上的
Pod 可能被驱逐或新 Pod 无法调度到该节点。

### 根因
节点控制器持续监控节点健康状态，默认每 5 秒检查一次。当节点变得不可达
（网络故障、kubelet 进程异常、节点资源耗尽等）时，节点控制器会把该节点的
`Ready` 状况更新为 `Unknown`。若节点持续保持不可达状态，控制器会在标记为
`Unknown` 后等待 5 分钟，然后触发该节点上全部 Pod 的驱逐流程。

### 处理步骤
1. 执行 `kubectl describe node <NODE_NAME>`，查看 `Conditions` 区域的
   `Ready` 状况和最近一次心跳时间（`LastHeartbeatTime`）
2. 登录该节点检查 kubelet 进程是否存活（`systemctl status kubelet`），
   以及节点本身的网络连通性
3. 检查节点资源使用情况（磁盘空间、内存），资源耗尽会导致 kubelet 无法
   正常上报心跳
4. 若节点已经离线且预计短期无法恢复，考虑手动执行驱逐流程，避免这批 Pod
   长期停留在不可用节点上不被重新调度

---
来源：[Kubernetes 官方文档 - 调试 Pod](https://kubernetes.io/zh-cn/docs/tasks/debug/debug-application/debug-pods/)、
[节点](https://kubernetes.io/zh-cn/docs/concepts/architecture/nodes/)，
CC BY 4.0 授权，节选整理。
