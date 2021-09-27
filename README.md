# ICE6405P-260-M01

Course materials

## Slides

| No. | Course Title | Slide |
| ---------- | ------------ | ----- |
| 1 | Cloud computing | [cloud-computing.pdf](./slides/cloud-computing.pdf) |
| 2 | Virtualization techniques | [virtualization.pdf](./slides/cloud-computing.pdf) |

## Labs

| No. | Labs |
| --- | ----------- |
| 1 | [Virtualization](./labs/Virtualization-Lab.docx) |
| 2 | [Serverless](./labs/Serverless-Labs.docx) |
| 3 | [Federated learning](./labs/Federated-Learning-Lab.docx) |

## Paper list

### Virtualization

&nbsp;&nbsp;&nbsp;&nbsp;  **I/O Virtualization**

* Dong, Yaozu, et al. "High performance network virtualization with SR-IOV." Journal of Parallel and Distributed Computing 72.11 (2012): 1471-1480.
* Xu, Xin, and Bhavesh Davda. "Srvm: Hypervisor support for live migration with passthrough sr-iov network devices." ACM SIGPLAN Notices 51.7 (2016): 65-77.
* Gordon, Abel, et al. "Towards exitless and efficient paravirtual I/O." Proceedings of the 5th Annual International Systems and Storage Conference. 2012.
* Zhang, Xiantao, et al. "High-density Multi-tenant Bare-metal Cloud." Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems. 2020.
* Peng, Bo, et al. "MDev-NVMe: A NVMe storage virtualization solution with mediated pass-through." 2018 {USENIX} Annual Technical Conference ({USENIX}{ATC} 18). 2018.
* Hildenbrand, David, and Martin Schulz. "virtio-mem: paravirtualized memory hot (un) plug." Proceedings of the 17th ACM SIGPLAN/SIGOPS International Conference on Virtual Execution Environments. 2021.

&nbsp;&nbsp;&nbsp;&nbsp;  **Accelerator**

* Hu, Xiaokang, et al. "{QZFS}:{QAT} Accelerated Compression in File System for Application Agnostic and Cost Efficient Data Storage." 2019 {USENIX} Annual Technical Conference ({USENIX}{ATC} 19). 2019.
* Li, Jian, et al. "QWEB: high-performance event-driven web architecture with QAT acceleration." IEEE Transactions on Parallel and Distributed Systems 31.11 (2020): 2633-2649.

&nbsp;&nbsp;&nbsp;&nbsp;  **GPU Virtualization**

* Lu, Qiumin, et al. "gMig: Efficient vGPU Live Migration with Overlapped Software-Based Dirty Page Verification." IEEE Transactions on Parallel and Distributed Systems 31.5 (2019): 1209-1222.
* Dong, Yaozu, et al. "Boosting {GPU} Virtualization Performance with Hybrid Shadow Page Tables." 2015 {USENIX} Annual Technical Conference ({USENIX}{ATC} 15). 2015.

### Serverless

&nbsp;&nbsp;&nbsp;&nbsp;  **Survey**

* Jonas, Eric, et al. "Cloud programming simplified: A berkeley view on serverless computing." arXiv preprint arXiv:1902.03383 (2019).

&nbsp;&nbsp;&nbsp;&nbsp;  **System Design**

* Akkus, Istemi Ekin, et al. "{SAND}: Towards High-Performance Serverless Computing." 2018 {Usenix} Annual Technical Conference ({USENIX}{ATC} 18). 2018.
* Oakes, Edward, et al. "{SOCK}: Rapid task provisioning with serverless-optimized containers." 2018 {USENIX} Annual Technical Conference ({USENIX}{ATC} 18). 2018.
* Agache, Alexandru, et al. "Firecracker: Lightweight virtualization for serverless applications." 17th {usenix} symposium on networked systems design and implementation ({nsdi} 20). 2020.

&nbsp;&nbsp;&nbsp;&nbsp;  **Cold Start**

* Shahrad, Mohammad, et al. "Serverless in the wild: Characterizing and optimizing the serverless workload at a large cloud provider." 2020 {USENIX} Annual Technical Conference ({USENIX}{ATC} 20). 2020.
* Du, Dong, et al. "Catalyzer: Sub-millisecond startup for serverless computing with initialization-less booting." Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems. 2020.
* Cadden, James, et al. "SEUSS: skip redundant paths to make serverless fast." Proceedings of the Fifteenth European Conference on Computer Systems. 2020.

&nbsp;&nbsp;&nbsp;&nbsp;  **Storage**

* Klimovic, Ana, et al. "Pocket: Elastic ephemeral storage for serverless analytics." 13th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 18). 2018.
* Zhang, Tian, et al. "Narrowing the gap between serverless and its state with storage functions." Proceedings of the ACM Symposium on Cloud Computing. 2019.

&nbsp;&nbsp;&nbsp;&nbsp;  **Benchmarking**

* Shahrad, Mohammad, Jonathan Balkind, and David Wentzlaff. "Architectural implications of function-as-a-service computing." Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture. 2019.
* Yu, Tianyi, et al. "Characterizing serverless platforms with serverlessbench." Proceedings of the 11th ACM Symposium on Cloud Computing. 2020.

&nbsp;&nbsp;&nbsp;&nbsp;  **Machine Learning**
* Thorpe, John, et al. "Dorylus: affordable, scalable, and accurate GNN training with distributed CPU servers and serverless threads." 15th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 21). 2021.
* Wang, Hao, Di Niu, and Baochun Li. "Distributed machine learning with a serverless architecture." IEEE INFOCOM 2019-IEEE Conference on Computer Communications. IEEE, 2019.
* Zhang, Chengliang, et al. "Mark: Exploiting cloud services for cost-effective, slo-aware machine learning inference serving." 2019 {USENIX} Annual Technical Conference ({USENIX}{ATC} 19). 2019.
* Ali, Ahsan, et al. "Batch: machine learning inference serving on serverless platforms with adaptive batching." SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2020.

### Federated Learning

&nbsp;&nbsp;&nbsp;&nbsp;  **Core**

* Konečný, Jakub, Brendan McMahan, and Daniel Ramage. "Federated optimization: Distributed optimization beyond the datacenter." arXiv preprint arXiv:1511.03575 (2015).
* McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.
* Konečný, Jakub, et al. "Federated learning: Strategies for improving communication efficiency." arXiv preprint arXiv:1610.05492 (2016).
* Li, Tian, et al. "Federated learning: Challenges, methods, and future directions." IEEE Signal Processing Magazine 37.3 (2020): 50-60.
* Mohri, Mehryar, Gary Sivek, and Ananda Theertha Suresh. "Agnostic federated learning." International Conference on Machine Learning. PMLR, 2019.

&nbsp;&nbsp;&nbsp;&nbsp;  **Algorithm**

* Wang, Hongyi, et al. "Federated Learning with Matched Averaging." International Conference on Learning Representations. 2019.
* Nishio, Takayuki, and Ryo Yonetani. "Client selection for federated learning with heterogeneous resources in mobile edge." ICC 2019-2019 IEEE International Conference on Communications (ICC). IEEE, 2019.
* Li, Tian, et al. "Federated optimization in heterogeneous networks." arXiv preprint arXiv:1812.06127 (2018).
* Shoham, Neta, et al. "Overcoming forgetting in federated learning on non-iid data." arXiv preprint arXiv:1910.07796 (2019).
* Smith, Virginia, et al. "Federated multi-task learning." Proceedings of the 31st International Conference on Neural Information Processing Systems. 2017.

&nbsp;&nbsp;&nbsp;&nbsp;  **Transformation**

* van Berlo, Bram, Aaqib Saeed, and Tanir Ozcelebi. "Towards federated unsupervised representation learning." Proceedings of the Third ACM International Workshop on Edge Systems, Analytics and Networking. 2020.

&nbsp;&nbsp;&nbsp;&nbsp;  **Application**

* Yang, Timothy, et al. "Applied federated learning: Improving google keyboard query suggestions." arXiv preprint arXiv:1812.02903 (2018).

&nbsp;&nbsp;&nbsp;&nbsp;  **Deployment**

* Bonawitz, Keith, et al. "Towards federated learning at scale: System design." arXiv preprint arXiv:1902.01046 (2019).

&nbsp;&nbsp;&nbsp;&nbsp;  **Security**

* Wu, Zhaoxian, et al. "Federated variance-reduced stochastic gradient descent with robustness to byzantine attacks." IEEE Transactions on Signal Processing 68 (2020): 4583-4596.
* Bagdasaryan, Eugene, et al. "How to backdoor federated learning." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
* Sun, Ziteng, et al. "Can you really backdoor federated learning?." arXiv preprint arXiv:1911.07963 (2019).
* Hanxi, Guo et al. "Siren: Byzantine-robust Federated Learning via Proactive Alarming."


## Contacts

* **Instructor**: Ruhui Ma - SEIEE 03-229
* **Teaching assistant**: Jianqing Zhang - SEIEE 03-309  Email: tsingz@sjtu.edu.cn
