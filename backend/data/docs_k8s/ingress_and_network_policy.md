# Ingress 与网络策略故障排查

## Ingress 配置后无法访问服务

### 现象
创建了 Ingress 资源，规则看起来配置正确，但通过域名/路径访问时得不到预期
响应，或者 Ingress 的 `ADDRESS` 字段长期显示 `<pending>`。

### 根因
最常见的根因是集群里没有实际运行 Ingress 控制器——仅创建 Ingress 资源本身
不会产生任何效果，必须有一个 Ingress 控制器负责监听并落地这些规则。另外，
`ADDRESS` 分配需要 Ingress 控制器和负载均衡器配合完成，通常需要一到两分钟，
这段时间内看到 `<pending>` 是正常现象而非故障。若规则本身没有匹配到任何
路径，流量会被路由到 `defaultBackend`（如果配置了的话），否则具体行为由
所选的 Ingress 控制器决定。

### 处理步骤
1. 确认集群中确实部署了 Ingress 控制器（`kubectl get pods -n <ingress控制器所在命名空间>`），
   仅有 Ingress 资源、没有控制器等于配置不会生效
2. 若 `ADDRESS` 长期是 `<pending>`（超过几分钟），检查 Ingress 控制器自身的
   Pod 状态和日志，以及云负载均衡器的分配情况
3. 核对 Ingress 规则里的 `host`/`path` 是否与实际请求的域名和路径完全匹配，
   注意路径匹配类型（`Exact`/`Prefix`）的区别，前缀不匹配的路径会退回
   `defaultBackend` 或直接得不到响应
4. 确认规则里引用的后端 Service 确实存在，且该 Service 本身可以正常访问
   （先排除 Service 自身的 Endpoints 问题，再排查 Ingress 层）

## NetworkPolicy 生效后 Pod 间无法通信

### 现象
给某个命名空间或某批 Pod 配置了 NetworkPolicy 之后，原本能正常通信的 Pod
之间突然连不上了，或者某个 Pod 无法再访问外部服务。

### 根因
NetworkPolicy 的隔离规则是叠加生效的：默认情况下 Pod 的出入流量都是非隔离
（全部允许）状态；一旦有任何 NetworkPolicy 的 `podSelector` 选中了该 Pod
并在 `policyTypes` 里声明了 `Ingress` 或 `Egress`，对应方向就会从"不限制"
切换为"只允许被规则明确允许的连接"。这意味着即使只是想"新增一条允许规则"，
只要该 NetworkPolicy 选中了这个 Pod，其他没有被任何规则覆盖的连接也会被
一并拒绝——很容易出现"只想开个口子，结果误关掉了其他本该能通的连接"的情况。

### 处理步骤
1. 执行 `kubectl get networkpolicy -n <命名空间>` 列出该命名空间下全部
   NetworkPolicy，逐条检查 `podSelector` 是否选中了受影响的 Pod
2. 对每条选中了该 Pod 的策略，检查其 `policyTypes` 声明的方向
   （`Ingress`/`Egress`），以及对应方向的规则列表是否覆盖了当前被拒绝的连接
3. 特别注意"默认拒绝所有流量"类策略（`podSelector: {}` 且不写任何
   `ingress`/`egress` 规则）——这类策略会隔离命名空间下的全部 Pod，即使
   这些 Pod 没有被其他策略单独选中过，也会被这条策略隐式覆盖
4. 需要放行某类连接时，新增一条明确允许该来源/目的的规则，而不是删除已有
   的默认拒绝策略——删除默认拒绝会让整个命名空间的隔离失效
5. 注意 `deny all` 类策略只保证拦截 TCP/UDP/SCTP，对 ICMP 等协议的行为
   未定义，排查连通性问题时不要假设 ping 不通就等于 TCP 也不通（反之亦然）

---
来源：[Kubernetes 官方文档 - Ingress](https://kubernetes.io/zh-cn/docs/concepts/services-networking/ingress/)、
[网络策略](https://kubernetes.io/zh-cn/docs/concepts/services-networking/network-policies/)，
CC BY 4.0 授权，节选整理。
