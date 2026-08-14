# RBAC 权限与资源配额故障排查

## kubectl 操作返回 Forbidden（RBAC 权限不足）

### 现象
用户或 ServiceAccount 执行 `kubectl` 命令（如查看/创建某类资源）时返回
`Forbidden`，提示当前身份没有权限执行该操作。

### 根因
RBAC 通过 `Role`/`ClusterRole`（定义一组权限规则）+ `RoleBinding`/
`ClusterRoleBinding`（把规则绑定给用户/组/ServiceAccount）两层对象组合生效。
`Role` 的权限范围限定在创建它时指定的命名空间内；`ClusterRole` 是集群范围，
不属于任何命名空间。`RoleBinding` 可以引用同命名空间的 `Role`，也可以引用
`ClusterRole` 但只在该 RoleBinding 所在命名空间生效；只有 `ClusterRoleBinding`
才能让 `ClusterRole` 的权限在整个集群范围生效。RBAC 的权限规则是纯粹累加的，
不存在"拒绝规则"——如果没有任何规则明确允许某个操作，默认就是拒绝。

### 官方语义边界
Kubernetes 官方 RBAC 文档说明，RBAC 策略通过 Kubernetes API 动态配置；
`RoleBinding`/`ClusterRoleBinding` 向其 subjects 授予所引用角色定义的权限。
修改绑定后，应使用 `kubectl auth can-i <verb> <resource> --as=<用户或 ServiceAccount>`
验证该身份的实际授权结果。本文档不对配置传播时延或任何组件是否需要重启作额外承诺。

### 处理步骤
1. 执行 `kubectl auth can-i <verb> <resource> --as=<用户或 ServiceAccount>`
   确认该身份对目标资源确实缺少权限，排除是操作本身写错的可能
2. 查找与该身份关联的全部 `RoleBinding`/`ClusterRoleBinding`
   （`kubectl get rolebinding,clusterrolebinding -A -o wide` 后按用户名/
   ServiceAccount 名过滤），确认是否真的绑定了预期的角色
3. 检查关联的 `Role`/`ClusterRole` 里是否包含目标操作所需的 `verb`
   （如 `get`/`list`/`create`/`delete`）和 `resource`/`apiGroup`
4. 若操作的是命名空间级资源但只找到了集群范围的 `ClusterRoleBinding`，
   或反过来只有命名空间内的 `RoleBinding` 却想操作集群范围资源，需要确认
   绑定类型和资源类型的范围是否匹配——范围不匹配时即使角色定义本身没问题，
   权限也不会生效
5. 排查时可以把 API 服务器日志级别调到 5（`--vmodule=rbac*=5`），日志里
   带 `RBAC` 前缀的行会记录具体是哪次鉴权被拒绝，比逐条排查绑定关系更直接

## 命名空间资源配额超限（ResourceQuota）

### 现象
在命名空间内创建 Pod、Service 或 PersistentVolumeClaim 等资源时被拒绝，
错误信息里包含 `exceeded quota`。

### 根因
命名空间配置了 `ResourceQuota` 对象，对该命名空间内某类资源的数量或总资源量
（CPU/内存请求与上限）设置了硬性上限。当尝试创建的资源会导致某一项配额超出
`hard` 字段声明的上限时，API 服务器会直接拒绝这次创建请求，不会进行部分创建。

### 处理步骤
1. 执行 `kubectl get resourcequota -n <命名空间> -o yaml` 查看该命名空间下
   全部配额对象的 `hard`（上限）和 `used`（当前已用量）
2. 对照错误信息里提到的具体资源类型（如 `persistentvolumeclaims`、
   `services.loadbalancers`），确认是哪一项配额已经用满
3. 若确实需要更多配额：清理该命名空间内闲置/多余的同类资源释放配额，
   或者与集群管理员协商调高该命名空间的 `ResourceQuota` 上限
4. 若配额刚好被占满但看不出哪些资源占用了它，注意配额既可能统计整个命名空间
   的资源总量，也可能是按具体对象类型分别计数——需要对照 `ResourceQuota`
   定义里具体统计的资源名称，而不是笼统地认为"配额不够了"

---
来源：[Kubernetes 官方文档 - 使用 RBAC 鉴权](https://kubernetes.io/zh-cn/docs/reference/access-authn-authz/rbac/)、
[为 API 对象配置配额](https://kubernetes.io/zh-cn/docs/tasks/administer-cluster/quota-api-object/)，
CC BY 4.0 授权，节选整理。
