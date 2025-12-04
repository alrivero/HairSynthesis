## `scontrol show partition`
```bash
PartitionName=a100
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=a100-qos
   DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=2 MaxTime=08:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=64
   Nodes=a100-[01-11]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=YES:4
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=704 TotalNodes=11 SelectTypeParameters=NONE
   JobDefaults=DefCpuPerGPU=16,DefMemPerGPU=4016
   DefMemPerCPU=4016 MaxMemPerCPU=257024

PartitionName=a100-large
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=a100-large-qos
   DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=4 MaxTime=08:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=64
   Nodes=a100-[01-11]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=YES:4
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=704 TotalNodes=11 SelectTypeParameters=NONE
   JobDefaults=DefCpuPerGPU=16,DefMemPerGPU=4016
   DefMemPerCPU=4016 MaxMemPerCPU=257024

PartitionName=a100-long
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=a100-long-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=1 MaxTime=2-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=64
   Nodes=a100-[01-11]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=YES:4
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=704 TotalNodes=11 SelectTypeParameters=NONE
   JobDefaults=DefCpuPerGPU=16,DefMemPerGPU=4016
   DefMemPerCPU=4016 MaxMemPerCPU=257024

PartitionName=debug-40core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=debug-40core-qos
   DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=6 MaxTime=01:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dn[001-030,032-064],rn[001-006]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2800 TotalNodes=69 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=extended-40core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=extended-40-core-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=2 MaxTime=7-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dn[001-030,032-064],rn[001-006]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2800 TotalNodes=69 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=extended-40core-shared
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=extended-40core-shared-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=1 MaxTime=3-12:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=40
   Nodes=dn[001-030,032-064],rn[001-006]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=YES:4
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2800 TotalNodes=69 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=2400 MaxMemPerCPU=161000

PartitionName=extended-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=extended-96-core-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=2 MaxTime=7-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dg[001-048]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=4608 TotalNodes=48 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=extended-96core-shared
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=extended-96core-shared-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=1 MaxTime=3-12:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=96
   Nodes=dg[001-048]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=YES:4
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=4608 TotalNodes=48 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=2400 MaxMemPerCPU=230400

PartitionName=hbm-1tb-long-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=1tb-hbm-long-96core-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=1 MaxTime=2-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=xm[045-048]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=384 TotalNodes=4 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=hbm-extended-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=hbm-extended-96core-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=2 MaxTime=7-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=xm[001-044,049-094]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=8640 TotalNodes=90 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=hbm-large-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=hbm-large-96core-qos
   DefaultTime=04:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=38 MaxTime=08:00:00 MinNodes=16 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=xm[001-044,049-094]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=8640 TotalNodes=90 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=hbm-long-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=hbm-long-96core-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=6 MaxTime=2-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=xm[001-044,049-094]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=8640 TotalNodes=90 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=hbm-medium-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=hbm-medium-96core-qos
   DefaultTime=04:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=16 MaxTime=12:00:00 MinNodes=6 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=xm[001-044,049-094]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=8640 TotalNodes=90 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=hbm-short-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=hbm-short-96core-qos
   DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=8 MaxTime=04:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=xm[001-044,049-094]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=8640 TotalNodes=90 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=large-40core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=large-40-core-qos
   DefaultTime=04:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=50 MaxTime=08:00:00 MinNodes=16 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dn[001-030,032-064],rn[001-006]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2800 TotalNodes=69 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=large-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=large-96-core-qos
   DefaultTime=04:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=38 MaxTime=08:00:00 MinNodes=16 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dg[001-048]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=4608 TotalNodes=48 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=long-40core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=long-40-core-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=6 MaxTime=2-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dn[001-030,032-064],rn[001-006]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2800 TotalNodes=69 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=long-40core-shared
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=long-40core-shared-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=3 MaxTime=1-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=40
   Nodes=dn[001-030,032-064],rn[001-006]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=YES:4
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2800 TotalNodes=69 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=2400 MaxMemPerCPU=161000

PartitionName=long-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=long-96-core-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=6 MaxTime=2-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dg[001-048]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=4608 TotalNodes=48 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=long-96core-shared
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=long-96core-shared-qos
   DefaultTime=08:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=3 MaxTime=1-00:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=96
   Nodes=dg[001-048]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=YES:4
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=4608 TotalNodes=48 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=2400 MaxMemPerCPU=230400

PartitionName=medium-40core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=medium-40-core-qos
   DefaultTime=04:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=16 MaxTime=12:00:00 MinNodes=6 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dn[001-030,032-064],rn[001-006]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2800 TotalNodes=69 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=medium-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=medium-96-core-qos
   DefaultTime=04:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=16 MaxTime=12:00:00 MinNodes=6 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dg[001-048]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=4608 TotalNodes=48 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=short-40core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=short-40-core-qos
   DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=8 MaxTime=04:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dn[001-030,032-064],rn[001-006]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2800 TotalNodes=69 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=short-40core-shared
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=short-40core-shared-qos
   DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=4 MaxTime=04:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=40
   Nodes=dn[001-030,032-064],rn[001-006]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=YES:4
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2800 TotalNodes=69 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=2400 MaxMemPerCPU=161000

PartitionName=short-96core
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=short-96-core-qos
   DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=8 MaxTime=04:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=dg[001-048]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=EXCLUSIVE
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=4608 TotalNodes=48 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

PartitionName=short-96core-shared
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=short-96core-shared-qos
   DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=4 MaxTime=04:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=96
   Nodes=dg[001-048]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=YES:4
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=4608 TotalNodes=48 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=2400 MaxMemPerCPU=230400
```