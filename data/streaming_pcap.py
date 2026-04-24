"""
NetSentinel-RL — Live pcap/Zeek ingestion via Scapy
Converts live packets into flow feature vectors for real-time inference.
"""
import time, threading, queue
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import logging

log = logging.getLogger(__name__)

@dataclass
class FlowKey:
    src_ip: str; dst_ip: str; src_port: int; dst_port: int; protocol: int
    def __hash__(self): return hash((self.src_ip,self.dst_ip,self.src_port,self.dst_port,self.protocol))

@dataclass
class FlowRecord:
    key: FlowKey
    start_time: float = field(default_factory=time.time)
    fwd_packets: int=0; bwd_packets: int=0
    fwd_bytes: int=0;   bwd_bytes: int=0
    fwd_pkt_lens: list=field(default_factory=list)
    bwd_pkt_lens: list=field(default_factory=list)
    fwd_iats: list=field(default_factory=list)
    bwd_iats: list=field(default_factory=list)
    syn_count: int=0; rst_count: int=0; ack_count: int=0
    psh_count: int=0; urg_count: int=0
    last_fwd_time: float=0; last_bwd_time: float=0
    all_iats: list=field(default_factory=list)
    last_pkt_time: float=field(default_factory=time.time)

    def to_feature_dict(self):
        dur=(time.time()-self.start_time)*1000
        total_pkts=self.fwd_packets+self.bwd_packets
        eps=1e-9
        fwd_lens=self.fwd_pkt_lens or [0]
        bwd_lens=self.bwd_pkt_lens or [0]
        all_iats=self.all_iats or [0]
        return {
            "Flow Duration":dur,
            "Total Fwd Packets":self.fwd_packets,
            "Total Backward Packets":self.bwd_packets,
            "Total Length of Fwd Packets":self.fwd_bytes,
            "Fwd Packet Length Max":max(fwd_lens),
            "Fwd Packet Length Mean":np.mean(fwd_lens),
            "Bwd Packet Length Max":max(bwd_lens),
            "Flow Bytes/s":(self.fwd_bytes+self.bwd_bytes)/(dur/1000+eps),
            "Flow Packets/s":total_pkts/(dur/1000+eps),
            "Flow IAT Mean":np.mean(all_iats),
            "Flow IAT Std":np.std(all_iats),
            "Fwd IAT Mean":np.mean(self.fwd_iats) if self.fwd_iats else 0,
            "Bwd IAT Mean":np.mean(self.bwd_iats) if self.bwd_iats else 0,
            "Fwd PSH Flags":self.psh_count,
            "Bwd PSH Flags":0,
            "Fwd URG Flags":self.urg_count,
            "SYN Flag Count":self.syn_count,
            "RST Flag Count":self.rst_count,
            "ACK Flag Count":self.ack_count,
            "Down/Up Ratio":self.bwd_bytes/(self.fwd_bytes+eps),
            "Avg Packet Size":(self.fwd_bytes+self.bwd_bytes)/(total_pkts+eps),
        }


class LiveFlowCollector:
    """
    Captures live packets with Scapy, assembles bidirectional flows,
    and emits completed flows to a queue for real-time inference.
    Timeout-based flow expiry (default 30s idle, 120s max).
    """
    IDLE_TIMEOUT=30; MAX_DURATION=120

    def __init__(self, interface="eth0", flow_queue: Optional[queue.Queue]=None):
        self.interface=interface
        self.flow_queue=flow_queue or queue.Queue(maxsize=1000)
        self.active_flows: Dict[FlowKey, FlowRecord]={}
        self._lock=threading.Lock()
        self._running=False

    def start(self):
        try:
            from scapy.all import sniff
            self._running=True
            threading.Thread(target=self._expire_loop, daemon=True).start()
            log.info(f"Starting capture on {self.interface}")
            sniff(iface=self.interface, prn=self._process_packet, store=False,
                  stop_filter=lambda _: not self._running)
        except ImportError:
            log.error("Scapy not installed. pip install scapy")
        except PermissionError:
            log.error("Root/sudo required for live capture.")

    def stop(self): self._running=False

    def _process_packet(self, pkt):
        try:
            from scapy.all import IP,TCP,UDP
            if not pkt.haslayer(IP): return
            ip=pkt[IP]; now=time.time()
            proto=6 if pkt.haslayer(TCP) else (17 if pkt.haslayer(UDP) else ip.proto)
            sport=pkt.sport if hasattr(pkt,"sport") else 0
            dport=pkt.dport if hasattr(pkt,"dport") else 0
            fwd_key=FlowKey(ip.src,ip.dst,sport,dport,proto)
            bwd_key=FlowKey(ip.dst,ip.src,dport,sport,proto)
            with self._lock:
                if fwd_key in self.active_flows:
                    flow=self.active_flows[fwd_key]; is_fwd=True
                elif bwd_key in self.active_flows:
                    flow=self.active_flows[bwd_key]; is_fwd=False
                else:
                    flow=FlowRecord(key=fwd_key); self.active_flows[fwd_key]=flow; is_fwd=True
                pkt_len=len(pkt); iat=(now-flow.last_pkt_time)*1000
                flow.all_iats.append(iat); flow.last_pkt_time=now
                if is_fwd:
                    flow.fwd_packets+=1; flow.fwd_bytes+=pkt_len
                    flow.fwd_pkt_lens.append(pkt_len)
                    if flow.last_fwd_time>0: flow.fwd_iats.append((now-flow.last_fwd_time)*1000)
                    flow.last_fwd_time=now
                    if pkt.haslayer(TCP):
                        flags=pkt[TCP].flags
                        flow.syn_count+=int(bool(flags&0x02))
                        flow.rst_count+=int(bool(flags&0x04))
                        flow.ack_count+=int(bool(flags&0x10))
                        flow.psh_count+=int(bool(flags&0x08))
                        flow.urg_count+=int(bool(flags&0x20))
                else:
                    flow.bwd_packets+=1; flow.bwd_bytes+=pkt_len
                    flow.bwd_pkt_lens.append(pkt_len)
                    if flow.last_bwd_time>0: flow.bwd_iats.append((now-flow.last_bwd_time)*1000)
                    flow.last_bwd_time=now
        except Exception as e:
            log.debug(f"Packet processing error: {e}")

    def _expire_loop(self):
        while self._running:
            time.sleep(5)
            now=time.time()
            with self._lock:
                expired=[]
                for key,flow in self.active_flows.items():
                    idle=(now-flow.last_pkt_time)
                    total=(now-flow.start_time)
                    if idle>self.IDLE_TIMEOUT or total>self.MAX_DURATION:
                        expired.append(key)
                for key in expired:
                    flow=self.active_flows.pop(key)
                    if flow.fwd_packets+flow.bwd_packets>=4:
                        feat=flow.to_feature_dict()
                        try: self.flow_queue.put_nowait(feat)
                        except queue.Full: pass

    def get_flow(self, timeout=1.0):
        try: return self.flow_queue.get(timeout=timeout)
        except queue.Empty: return None
