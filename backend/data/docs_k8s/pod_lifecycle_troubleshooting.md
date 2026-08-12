# Pod 生命周期故障排查

## Pod 停滞在 Pending 状态

### 现象
Pod 创建后长时间停留在 `Pending` 状态，一直没有进入 `Running`。

### 根因
`Pending` 表示 Pod 还没有被调度到任何节点上。通常是某种资源不足导致调度器
无法为其找到合适的节点，或者 Pod 使用了 `hostPort` 导致可调度的节点范围受限。

### 处理步骤
1. 执行 `kubectl describe pod <POD_NAME>`，查看 Events 区域调度器给出的原因
2. 若原因是资源不足：检查集群 CPU/内存剩余容量，删除闲置 Pod、调低资源请求，
   或为集群新增节点
3. 若 Pod 定义中使用了 `hostPort`：确认是否真的需要绑定主机端口——多数场景
   用 Service 暴露 Pod 即可，`hostPort` 会把可调度节点数限制为持有该端口的
   节点数量

## Pod 停滞在 Waiting 状态（镜像拉取失败）

### 现象
Pod 已经被调度到某个节点，但容器状态显示 `Waiting`，一直无法进入 `Running`，
`kubectl describe` 里能看到 `ImagePullBackOff` 或 `ErrImagePull`。

### 根因
`ImagePullBackOff` 说明 kubelet 反复尝试拉取镜像但失败，常见原因是镜像名称
拼写错误、镜像未推送到仓库、或从私有仓库拉取时缺少 `imagePullSecret`。
`BackOff` 表示 Kubernetes 会持续重试，重试间隔逐次增加，最长间隔 300 秒（5 分钟）。

### 处理步骤
1. 核对 Pod 定义里的镜像名称是否拼写正确（包括仓库地址、标签）
2. 确认镜像确实已经推送到目标仓库
3. 在本机手动执行一次镜像拉取（如 `docker pull <镜像>`）验证镜像本身可访问
4. 若目标是私有仓库，检查 Pod 是否正确配置了 `imagePullSecrets`——该字段
   引用的 Secret 必须存在于 Pod 所在的同一命名空间

## Pod 反复重启（CrashLoopBackOff）

### 现象
Pod 状态显示 `CrashLoopBackOff`，容器启动后很快退出，Kubernetes 反复尝试
重启但始终无法稳定运行。

### 根因
容器进程本身异常退出（应用报错、启动命令错误、健康检查连续失败等），
Kubernetes 检测到退出后按退避策略反复重启，重启间隔逐次拉长。

### 处理步骤
1. 执行 `kubectl describe pod <POD_NAME>`，确认 `Reason` 字段确实是
   `CrashLoopBackOff`，并查看 `Last State` 的退出码
2. 执行 `kubectl logs <POD_NAME> --previous`，查看上一次崩溃前的容器日志，
   而不是当前正在重启的容器
3. 若日志信息不够，可用 `kubectl debug` 创建一个共享该 Pod 网络/进程
   命名空间的调试容器进入现场排查
4. 常见根因包括：启动命令或参数写错、容器内进程主动退出、readiness/liveness
   探针配置过于激进导致被判定为不健康

## Pod 停滞在 Terminating 状态

### 现象
执行删除操作后，Pod 长时间停留在 `Terminating` 状态，一直没有真正被移除。

### 根因
Pod 上配置了 finalizer，同时集群里安装了针对 Pod `UPDATE` 操作的准入 Webhook
（`ValidatingWebhookConfiguration` 或 `MutatingWebhookConfiguration`），
该 Webhook 阻止了控制平面移除 finalizer，导致删除流程卡住。

### 处理步骤
1. 检查集群中是否存在针对 `pods` 资源 `UPDATE` 操作的 Webhook 配置
2. 若 Webhook 来自第三方组件：确认版本是否最新，尝试临时禁用其对 `UPDATE`
   操作的处理，并向对应组件提交问题反馈
3. 若 Webhook 是自己编写的：变更类 Webhook 不应修改不可变字段；验证类
   Webhook 的校验规则应只作用于新变更，不能拦截已存在的历史违规数据，
   否则会阻塞所有历史 Pod 的正常删除流程

---
来源：[Kubernetes 官方文档 - 调试 Pod](https://kubernetes.io/zh-cn/docs/tasks/debug/debug-application/debug-pods/)、
[镜像](https://kubernetes.io/zh-cn/docs/concepts/containers/images/)、
[调试运行中的 Pod](https://kubernetes.io/zh-cn/docs/tasks/debug/debug-application/debug-running-pod/)，
CC BY 4.0 授权，节选整理。
