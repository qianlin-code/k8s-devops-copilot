# 存储挂载与自动扩缩容故障排查

## PVC 一直停留在 Pending，Pod 无法挂载卷

### 现象
创建了 PersistentVolumeClaim（PVC），但它的状态长期停留在 `Pending`，
引用该 PVC 的 Pod 也因此无法启动，卡在 `ContainerCreating` 状态。

### 根因
PVC 要进入 `Bound` 状态，必须找到一个满足条件的 PersistentVolume（PV）——
两者的 `storageClassName` 必须一致（未设置 `storageClassName` 的 PV 只能
绑定不指定存储类的 PVC），并且 PV 的访问模式（`accessModes`）、容量必须
满足 PVC 的要求。PVC 与 PV 的绑定关系是排他性的一对一映射：一旦某个 PV 已
绑定给一个 PVC，其他 PVC 即使条件都满足也无法再绑定到这个 PV 上。如果集群
配置了动态供应（StorageClass 支持自动创建 PV），配额或后端存储系统的问题
也会导致动态创建失败，PVC 停留在 Pending。

### 处理步骤
1. 执行 `kubectl describe pvc <PVC_NAME>`，查看 Events 区域的具体报错信息——
   动态供应失败通常会在这里直接给出后端存储系统返回的错误
2. 若期望绑定到已有的静态 PV：执行 `kubectl get pv` 检查是否存在满足条件的
   可用 PV（`STATUS` 为 `Available`），核对 `storageClassName`、
   `accessModes`、容量是否与 PVC 的要求一致
3. 若期望走动态供应：确认对应的 `StorageClass` 存在且 `provisioner` 配置
   正确，检查该命名空间的 `ResourceQuota` 是否已经把
   `persistentvolumeclaims` 数量或存储容量配额用满
4. 挂载选项（`mountOptions`）不会在创建时做合法性校验，非法的挂载选项会让
   卷在真正挂载阶段才报错失败——如果 PVC 已经 `Bound` 但 Pod 仍然挂载失败，
   需要额外检查 PV 定义里的挂载选项是否正确

## HPA 配置后没有按预期扩缩容

### 现象
给 Deployment 配置了 HorizontalPodAutoscaler（HPA），但观察一段时间后副本数
始终没有变化，或者 `kubectl get hpa` 里 `TARGETS` 列显示 `<unknown>`。

### 根因
HPA 依赖 Metrics Server（或其他自定义/外部指标适配器）持续采集资源使用率
（CPU/内存等），如果集群没有部署 Metrics Server 或它工作异常，HPA 就完全
拿不到判断依据，`TARGETS` 会显示为 `<unknown>`，此时 HPA 不会做任何扩缩容
决策。另外，若 Pod 的容器没有配置资源请求（`resources.requests`），HPA
按 CPU/内存利用率百分比计算时也无法得出结果，因为利用率是相对于请求量算的。

### 处理步骤
1. 执行 `kubectl get hpa <HPA_NAME>`，若 `TARGETS` 列显示 `<unknown>`，先
   确认集群是否已部署 Metrics Server（`kubectl get deployment metrics-server -n kube-system`）
2. 确认 Metrics Server 本身处于健康运行状态，可用
   `kubectl top pods`/`kubectl top nodes` 验证能否正常取到资源使用数据——
   如果这两个命令本身都取不到数据，问题在 Metrics Server 而不是 HPA 配置
3. 检查目标 Deployment 的 Pod 是否配置了 `resources.requests`——没有设置
   请求量的容器，其资源利用率百分比无法被正确计算
4. 若指标一切正常但扩缩容行为看起来"反应迟钝"，检查 HPA 的
   `behavior`/稳定窗口配置（新版 API 中 `targetCPUUtilizationPercentage`
   已被更通用的 `metrics` 数组字段取代），HPA 默认有冷却时间避免抖动，
   短时间内的指标波动不会立即触发扩缩容

---
来源：[Kubernetes 官方文档 - 持久卷](https://kubernetes.io/zh-cn/docs/concepts/storage/persistent-volumes/)、
[基于 Pod 水平自动扩缩容示例](https://kubernetes.io/zh-cn/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/)，
CC BY 4.0 授权，节选整理。
