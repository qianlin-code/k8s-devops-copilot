# ConfigMap 引用故障排查

## Pod 因引用的 ConfigMap 不存在而无法启动

### 现象
Pod 一直无法进入 `Running` 状态，`kubectl describe pod` 里能看到
`CreateContainerConfigError` 一类的错误，或者容器内环境变量/挂载文件的值
是空的。

### 根因
Pod 规约里通过环境变量（`valueFrom.configMapKeyRef`）或卷
（`volumes.configMap`）引用某个 ConfigMap 时，Kubernetes 要求这个 ConfigMap
必须已经存在，除非在引用处显式标记为 `optional`。若引用的 ConfigMap 不存在
且没有标记 `optional`，Pod 无法启动；即使 ConfigMap 本身存在，但引用的具体
键（key）不存在，同样会导致 Pod 无法启动（除非该键引用也标记了 `optional`）。
如果是以卷方式挂载整个 ConfigMap，行为略有不同：ConfigMap 不存在时，挂载的
卷会是空目录；ConfigMap 存在但引用的键不存在时，对应路径下不会生成文件——
挂载路径本身总会被创建，只是内容为空，这一点容易被误判成"挂载失败"，实际
是"挂载了空内容"。

### 处理步骤
1. 执行 `kubectl describe pod <POD_NAME>`，确认报错是否为
   `CreateContainerConfigError`，并查看具体是哪个 ConfigMap/键引用出的问题
2. 执行 `kubectl get configmap <CONFIGMAP_NAME> -n <命名空间>` 确认该
   ConfigMap 确实存在于 Pod 所在的命名空间——ConfigMap 是命名空间级资源，
   跨命名空间引用会直接找不到
3. 若以环境变量方式引用具体键（`configMapKeyRef.key`），核对该键名是否与
   ConfigMap 里实际存在的键完全一致（大小写、拼写）
4. 若以卷方式挂载但发现挂载路径下文件为空而不是报错：这通常不是故障，而是
   ConfigMap 或对应键确实不存在的正常表现，需要回头检查 ConfigMap 内容本身
5. 若该引用在设计上允许缺失（比如可选配置项），给对应的
   `configMapKeyRef`/`configMap` 引用加上 `optional: true`，避免 ConfigMap
   缺失时连累整个 Pod 无法启动

---
来源：[Kubernetes 官方文档 - 配置 Pod 以使用 ConfigMap](https://kubernetes.io/zh-cn/docs/tasks/configure-pod-container/configure-pod-configmap/)，
CC BY 4.0 授权，节选整理。
