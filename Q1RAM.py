from enum import Enum
from typing import Optional,List
from qiskit import *
import numpy as np
from qiskit_aer import StatevectorSimulator,AerSimulator
from typing import List

class QramRegister(Enum):
    A_BUS = "QRAM_A_BUS"
    D_BUS = "QRAM_D_BUS"
    A = "QRAM_A"
    TOF_AUX = "QRAM_tof_aux"
    D = "QRAM_D"
    DQ = "QRAM_dq"
    R = "QRAM_r"
    CLASSICAL_ADDRESS = "QRAM_Address"
    CLASSICAL_DATA = "QRAM_Data"
    READ_FLAG = "READ_Flag"

from qiskit.circuit.instruction import Instruction
def Simplified_RTOF_Gate(num_controls:int,num_targets=1,clean_ancilla=True)->Instruction:
    qc= QuantumCircuit()
    num_ancilla=num_controls//2-1
    controls=QuantumRegister(num_controls,name="control")
    ancilla=QuantumRegister(num_ancilla,name="ancilla")
    target=QuantumRegister(num_targets,name="target")
    # qc.add_register(controls)
    # qc.add_register(ancilla)
    # qc.add_register(target)

    if(num_controls==3):
      qc.add_register(controls)
      qc.add_register(target)
      qc.rcccx(controls[0],controls[1],controls[2],target)
    elif(num_controls==2):
      qc.add_register(controls)
      qc.add_register(target)
      qc.rccx(controls[0],controls[1],target)
    else:
      qc.add_register(controls)
      qc.add_register(ancilla)
      qc.add_register(target)
      if(num_ancilla!=num_controls//2-1):
          raise ValueError(f"Expected {num_controls//2-1} ancilla qubits, while {num_ancilla} is provided")

      i=0
      num_remaining_controls= num_controls
      while (num_remaining_controls>0):
          if(i==0):
              # print(f"RTOF: c{i*2},c{i*2+1},c{i*2+2}-->a{i}")
              qc.rcccx(controls[i*2],controls[i*2+1],controls[i*2+2],ancilla[i])
              num_remaining_controls-=3
          elif(num_remaining_controls>2):
              # print(f"RTOF: a{i-1},c{i*2+1},c{i*2+2}-->a{i}")
              qc.rcccx(ancilla[i-1],controls[i*2+1],controls[i*2+2],ancilla[i])
              num_remaining_controls-=2
          elif(num_remaining_controls==2):
              # print(f"RTOF: a{i-1},c{i*2+1},c{i*2+2}-->tar")
              for t in range(num_targets):
                qc.rcccx(ancilla[i-1],controls[i*2+1],controls[i*2+2],target[t])
              num_remaining_controls-=2
          elif(num_remaining_controls==1):
              # print(f"ccx:  a{i-1},c{num_controls-1}-->tar")
              for t in range(num_targets):
                qc.rccx(ancilla[i-1],controls[num_controls-1],target[t])
              num_remaining_controls-=1
          i+=1
      if(clean_ancilla):
          i=0
          num_remaining_controls= num_controls
          qc_temp=qc.copy_empty_like()
          while (num_remaining_controls>2):
            if(i==0):
                qc_temp.rcccx(controls[i*2],controls[i*2+1],controls[i*2+2],ancilla[i])
                num_remaining_controls-=3
            elif(num_remaining_controls>2):
                qc_temp.rcccx(ancilla[i-1],controls[i*2+1],controls[i*2+2],ancilla[i])
                num_remaining_controls-=2
            i+=1
          qc_temp=qc_temp.inverse()
          qc.compose(qc_temp,inplace=True)

    qc.name=f"({num_controls})RTOF-{'CLEAN' if clean_ancilla else ''}"
    return qc.to_instruction()

def apply_log_depth_rtof(qc:QuantumCircuit,controls:QuantumRegister,ancilla:QuantumRegister,target:QuantumRegister,clean_ancilla=False):
    num_controls= len(controls)
    num_ancilla=len(ancilla)
    if(num_ancilla!=num_controls//2-1):
        raise ValueError(f"Expected {num_controls//2-1} ancilla qubits, while {num_ancilla} is provided")

    i=0
    num_remaining_controls= num_controls
    while (num_remaining_controls>0):
        if(i==0):
            # print(f"RTOF: c{i*2},c{i*2+1},c{i*2+2}-->a{i}")
            qc.rcccx(controls[i*2],controls[i*2+1],controls[i*2+2],ancilla[i])
            num_remaining_controls-=3
        elif(num_remaining_controls>2):
            # print(f"RTOF: a{i-1},c{i*2+1},c{i*2+2}-->a{i}")
            qc.rcccx(ancilla[i-1],controls[i*2+1],controls[i*2+2],ancilla[i])
            num_remaining_controls-=2
        elif(num_remaining_controls==2):
            # print(f"RTOF: a{i-1},c{i*2+1},c{i*2+2}-->tar")
            qc.rcccx(ancilla[i-1],controls[i*2+1],controls[i*2+2],target)
            num_remaining_controls-=2
        elif(num_remaining_controls==1):
            # print(f"ccx:  a{i-1},c{num_controls-1}-->tar")
            qc.rccx(ancilla[i-1],controls[num_controls-1],target)
            num_remaining_controls-=1
        i+=1
    if(clean_ancilla):
        i=0
        num_remaining_controls= num_controls
        qc_temp=qc.copy_empty_like()
        while (num_remaining_controls>2):
          if(i==0):
              qc_temp.rcccx(controls[i*2],controls[i*2+1],controls[i*2+2],ancilla[i])
              num_remaining_controls-=3
          elif(num_remaining_controls>2):
              qc_temp.rcccx(ancilla[i-1],controls[i*2+1],controls[i*2+2],ancilla[i])
              num_remaining_controls-=2
          i+=1

        qc_temp=qc_temp.inverse()
        qc.compose(qc_temp,inplace=True)

class Q1RAM:

    def __init__(self
                 ,addressLength
                 ,dataLength
                 ,qc:QuantumCircuit=None
                 ,qr_address_bus=None
                 ,qr_data_bus=None
                 ,qr_read_write=None
                 ,qr_memory_address=None
                 ,qr_memory_data=None
                 ,qr_tof_aux=None
                 ,use_simplified_toffoli=True):

        self.addressLength = addressLength
        self.dataLength = dataLength

        self.qrAddressRegister=qr_memory_address
        self.qrDataRegister=qr_memory_data
        self.use_external_tof_ancilla=False
        self.use_simplified_toffoli= use_simplified_toffoli
        self.current_data_bus_dirty=False

        # self.qrAddressRegister = QuantumRegister(self.addressLength, name=QramRegister.A.value)
        # self.qrDataRegister = QuantumRegister(self.dataLength, name=QramRegister.D.value)

        if(use_simplified_toffoli and addressLength>5):
            if(qr_tof_aux):
                self.qrToffoliAncilla= qr_tof_aux
                self.use_external_tof_ancilla=True
            else:
                self.qrToffoliAncilla = QuantumRegister((addressLength//2)-1, name=QramRegister.TOF_AUX.value)
        else:
            self.qrToffoliAncilla = None

        self.qrDirectionQubit = QuantumRegister(1, name=QramRegister.DQ.value)

        self.qrAddressBus=qr_address_bus
        self.qrDataBus=qr_data_bus
        self.qrReadWriteQubit=qr_read_write
        # self.qrReadFlagQubit=qr_read_flag


        self.crAddress = ClassicalRegister(addressLength, name=QramRegister.CLASSICAL_ADDRESS.value)
        self.crData = ClassicalRegister(dataLength, name=QramRegister.CLASSICAL_DATA.value)
        self.crReadFlag=ClassicalRegister(1,name="ReadFlag")
        self.qc= qc
        self.build()

    @property
    def qr_address_register_index(self):
      return [self.qc.qubits.index(qbit) for qbit in self.qrAddressRegister]

    @property
    def qr_data_register_index(self):
      return [self.qc.qubits.index(qbit) for qbit in self.qrDataRegister]

    @property
    def qr_toffoli_ancilla_index(self):  # Assuming qrToffoliAncilla exists
      if hasattr(self, 'qrToffoliAncilla'):
        return [self.qc.qubits.index(qbit) for qbit in self.qrToffoliAncilla]
      else:
        return []  # Return empty list if qrToffoliAncilla doesn't exist

    @property
    def qr_data_bus_index(self):
      return [self.qc.qubits.index(qbit) for qbit in self.qrDataBus]

    @property
    def qr_address_bus_index(self):
      return [self.qc.qubits.index(qbit) for qbit in self.qrAddressBus]


    @property
    def qr_direction_qubit_index(self):
      return self.qc.qubits.index(self.qrDirectionQubit[0])

    @property
    def qr_read_write_qubit_index(self):
      return self.qc.qubits.index(self.qrReadWriteQubit[0])



    def init_address_bus(self, bus: QuantumRegister=None):
        if bus:
            if len(bus) != self.addressLength:
                raise Exception(f"Invalid given address bus size, expected {self.addressLength}, given {len(bus)}")
            self.qrAddressBus = bus
        else:
            self.qrAddressBus = QuantumRegister(self.addressLength, name="A_BUS")
        self.registers.append(self.qrAddressBus)
        return self

    def init_data_bus(self, bus: QuantumRegister=None):
        if bus:
            if len(bus) != self.dataLength:
                raise Exception (f"Invalid given data bus size, expected {self.dataLength}, given {len(bus)}")
            self.qrDataBus = bus
        else:
            self.qrDataBus = QuantumRegister(self.dataLength, name="D_BUS")
        self.registers.append(self.qrDataBus)
        return self

    def init_read_write_qubit(self, qubit: QuantumRegister=None):
        if qubit:
            if len(qubit) != 1:
                raise Exception("Invalid given read/write qubit size, expected 1")
            self.qrReadWriteQubit = qubit
            self.registers.append(self.qrReadWriteQubit)
        return self

    def init_read_flag_qubit(self, qubit:QuantumRegister=None):
        if qubit:
            if len(qubit) != 1:
                raise Exception("Invalid given read flag qubit size, expected 1")
            self.qrReadFlagQubit = qubit
        else:
            self.qrReadFlagQubit = QuantumRegister(1, name="READ_FLAG")
        self.registers.append(self.qrReadFlagQubit)
        return self

    def init_memory_address(self,bus:QuantumRegister=None):
        if bus:
            if len(bus) != self.addressLength:
                raise Exception(f"Invalid given address bus size, expected {self.addressLength}, given {len(bus)}")
            self.qrAddressRegister = bus
        else:
            self.qrAddressRegister = QuantumRegister(self.addressLength, name="QRAM_A")
        self.registers.append(self.qrAddressRegister)

        return self

    def init_memory_data(self, bus: QuantumRegister=None):
        if bus:
            if len(bus) != self.dataLength:
                raise Exception (f"Invalid given data bus size, expected {self.dataLength}, given {len(bus)}")
            self.qrDataRegister = bus
        else:
            self.qrDataRegister = QuantumRegister(self.dataLength, name="QRAM_D")
        self.registers.append(self.qrDataRegister)
        return self


    def build(self):
        self.registers=[self.qrDirectionQubit]
        self.qc.add_register(*self.registers)
        # print(f"after build:{self.qc.qubits.index(self.qrDirectionQubit[0])}")
        if(self.qrAddressRegister is None):
            self.init_memory_address()
            self.qc.add_register(self.qrAddressRegister)
            self.qc.h(self.qrAddressRegister)
        else:
            self.init_memory_address(self.qrAddressRegister)


        if(self.qrDataRegister is None):
            self.init_memory_data()
            self.qc.add_register(self.qrDataRegister)
        else:
            self.init_memory_data(self.qrDataRegister)


        if(self.qrAddressBus is None):
            self.init_address_bus()
            self.qc.add_register(self.qrAddressBus)
        else:
            self.init_address_bus(self.qrAddressBus)


        if(self.qrDataBus is None):
            self.init_data_bus()
            self.qc.add_register(self.qrDataBus)
        else:
            self.init_data_bus(self.qrDataBus)



        if(self.qrToffoliAncilla and not self.use_external_tof_ancilla):
            self.qc.add_register(self.qrToffoliAncilla)

        # if(self.qrReadFlagQubit is None):
        #     pass
        #     # self.init_read_flag_qubit()
        #     # self.qc.add_register(self.qrReadFlagQubit)
        # else:
        #     self.init_read_flag_qubit(self.qrReadFlagQubit)

        if(self.qrReadWriteQubit):
            self.init_read_write_qubit(self.qrReadWriteQubit)
            self.qc.add_register(self.qrReadWriteQubit)


        # self.qc.h(self.qrAddressRegister)

    def apply_read(self):
        if(self.addressLength>4):
          raise Exception("Exceeded Usage Limitations")
        qc_copy= self.qc.copy_empty_like()
        qc_copy.name="Read"#f"QRAM {self.addressLength}.{self.dataLength}_read"
        self._read_only(qc_copy)
        ReadGate= qc_copy.to_instruction()
        self.qc.append(ReadGate,range(self.qc.num_qubits))#,[self.qc.qubits.index(qubit) for qreg in self.registers for qubit in qreg])

    def apply_write(self):
        if(self.addressLength>4):
          raise Exception("Exceeded Usage Limitations")
        qc_copy= self.qc.copy_empty_like()
        # qc_copy.name=f"QRAM {self.addressLength}.{self.dataLength}_write"
        qc_copy.name="Write"
        self._write_only(qc_copy)
        WriteGate= qc_copy.to_instruction()
        self.qc.append(WriteGate,range(self.qc.num_qubits))#,[self.qc.qubits.index(qubit) for qreg in self.registers for qubit in qreg])


    def apply_read_write(self):
        if(self.addressLength>4):
          raise Exception("Exceeded Usage Limitations")
        qc_copy= self.qc.copy_empty_like()
        qc_copy.name="1-Read/0-write"#f"QRAM {self.addressLength}.{self.dataLength}_Read_Write"
        self._read_write(qc_copy)
        ReadWriteGate= qc_copy.to_instruction()
        self.qc.append(ReadWriteGate,range(self.qc.num_qubits))#,[self.qc.qubits.index(qubit) for qreg in self.registers for qubit in qreg])

    def _forward(self,q_circ:QuantumCircuit=None):
        if(self.addressLength>4):
          raise Exception("Exceeded Usage Limitations")
        if(q_circ is None):
            q_circ= self.qc
        for i in range(self.addressLength):
            q_circ.cx(self.qr_address_bus_index[i], self.qr_address_register_index[i])
            q_circ.x(self.qr_address_register_index[i])
        if(self.addressLength<=5 or not self.use_simplified_toffoli):
            q_circ.mcx(self.qr_address_register_index, self.qr_direction_qubit_index)
        else:
            # apply_log_depth_rtof(q_circ,self.qr_address_register_index,self.qr_toffoli_ancilla_index,self.qr_direction_qubit_index,clean_ancilla=True)
            rtof_gate=Simplified_RTOF_Gate(num_controls=len(self.qr_address_register_index),clean_ancilla=True)
            q_circ.append(rtof_gate,self.qr_address_register_index+self.qr_toffoli_ancilla_index+[self.qr_direction_qubit_index])

    def _ccswap(self, ctrl1, ctrl2, targ1, targ2,q_circ:QuantumCircuit=None):
        if(q_circ is None):
            q_circ= self.qc
        q_circ.cx(targ2, targ1)
        q_circ.mcx([ctrl1, ctrl2, targ1],targ2)
        q_circ.cx(targ2, targ1)


    def _read_only(self,q_circ:QuantumCircuit=None):
        if(self.addressLength>4):
          raise Exception("Exceeded Usage Limitations")
        if(q_circ is None):
          q_circ= self.qc
        self._forward(q_circ)

        for i in range(self.dataLength):
            # q_circ.mcx([self.qr_direction_qubit_index,self.qr_data_register_index[i]],self.qr_data_bus_index[i])
            q_circ.rccx(self.qr_direction_qubit_index,self.qr_data_register_index[i],self.qr_data_bus_index[i])


        self._reverse(q_circ)

    def _write_only(self,q_circ:QuantumCircuit=None):
        if(self.addressLength>4):
          raise Exception("Exceeded Usage Limitations")
        if(q_circ is None):
          q_circ= self.qc
        self._forward(q_circ)

        for i in range(self.dataLength):
            # q_circ.cswap(self.qr_direction_qubit_index ,self.qr_data_register_index[i], self.qr_data_bus_index[i])
            q_circ.rccx(self.qr_direction_qubit_index ,self.qr_data_bus_index[i],self.qr_data_register_index[i])
        self._reverse(q_circ)

    def _read_write(self,q_circ:QuantumCircuit=None):
        if(self.addressLength>4):
          raise Exception("Exceeded Usage Limitations")
        if(q_circ is None):
          q_circ= self.qc
        q_circ.x(self.qr_read_write_qubit_index)

        for i in range(self.dataLength):
            # self._ccswap( self.qr_read_write_qubit_index,self.qr_direction_qubit_index ,self.qr_data_register_index[i], self.qr_data_bus_index[i] ,q_circ=q_circ)
            self.qc.rcccx( [self.qr_read_write_qubit_index,self.qr_direction_qubit_index ,self.qr_data_bus_index[i]],self.qr_data_register_index[i])
            # q_circ.barrier()
        q_circ.x(self.qr_read_write_qubit_index)
        for i in range(self.dataLength):
            q_circ.mcx([self.qr_read_write_qubit_index,self.qr_direction_qubit_index,self.qr_data_register_index[i]],self.qr_data_bus_index[i])
        # q_circ.barrier()

    def _reverse(self,q_circ:QuantumCircuit=None):
        if(q_circ is None):
          q_circ= self.qc

        if(self.addressLength<=5 or not self.use_simplified_toffoli):
            q_circ.mcx(self.qr_address_register_index, self.qr_direction_qubit_index)
        else:
            # apply_log_depth_rtof(q_circ,self.qr_address_register_index,self.qr_toffoli_ancilla_index,self.qr_direction_qubit_index,clean_ancilla=True)
            rtof_gate=Simplified_RTOF_Gate(num_controls=len(self.qr_address_register_index),clean_ancilla=True)
            q_circ.append(rtof_gate,self.qr_address_register_index+self.qr_toffoli_ancilla_index+[self.qr_direction_qubit_index])

        for i in range(self.addressLength):
            q_circ.x(self.qr_address_register_index[i])
            q_circ.cx(self.qr_address_bus_index[i], self.qr_address_register_index[i])

        # q_circ.barrier()


    import numpy as np

    def superposed_state_vector(self,indices, num_qubits):
      """
      Creates a normalized quantum state vector as a superposition of
      computational basis states given by the indices.

      Parameters:
      - indices (list of int): Positions where the state vector should have non-zero amplitudes.
      - num_qubits (int): Number of qubits (determines the vector size: 2^num_qubits).

      Returns:
      - np.ndarray: Normalized state vector.
      """
      dim = 2 ** num_qubits
      state = np.zeros(dim, dtype=np.complex128)

      for idx in indices:
          if not (0 <= idx < dim):
              raise ValueError(f"Index {idx} is out of bounds for {num_qubits} qubits.")
          state[idx] = 1.0

      norm = np.linalg.norm(state)
      if norm == 0:
          raise ValueError("Cannot normalize a zero vector (empty index list).")

      return state / norm



    def Read(self,address_state,normalize=False):
        # if(self.current_data_bus_dirty):
        #     qr_data_bus2= QuantumRegister(self.dataLength)
        #     self.qc.add_register(qr_data_bus2)
        #     self.qrDataBus= qr_data_bus2
        # if(len(address_state) >1):
        #     print(address_state)
        #     address_state=self.superposed_state_vector(address_state,self.addressLength)
        self.qc.initialize(address_state,self.qrAddressBus,normalize=normalize)
        # self.qc.initialize(0,self.qrDataBus)
        self.apply_read()
        self.current_data_bus_dirty=True

    def ReadAll(self,remove_null_state=False):
        # if(self.current_data_bus_dirty):
        #     qr_data_bus2= QuantumRegister(self.dataLength)
        #     self.qc.add_register(qr_data_bus2)
        #     self.qrDataBus= qr_data_bus2
        self.qc.reset(self.qrAddressBus)
        self.qc.reset(self.qrDataBus)
        self.qc.h(self.qrAddressBus)
        self.apply_read()
        self.current_data_bus_dirty=True
        # for i in range(self.addressLength):
        #   self.qc.ch(self.qrReadFlagQubit,self.qrAddressBus[i],"0")

    # def WriteAll(self,data):
    #     self.qc.initialize(0,self.qrAddressBus)
    #     self.qc.h(self.qrAddressBus)
    #     self.qc.prepare_state(data,self.qrDataBus)
    #     self.apply_write()

    def Write(self,address_state,data_state,normalize=False):
      if(self.addressLength>4 or self.dataLength>8):
        raise Exception("Exceeded Usage Limitations")
      # if(self.current_data_bus_dirty):
      #     qr_data_bus2= QuantumRegister(self.dataLength)
      #     self.qc.add_register(qr_data_bus2)
      #     self.qrDataBus= qr_data_bus2
      self.qc.initialize(address_state,self.qrAddressBus,normalize=normalize)
      self.qc.initialize(data_state,self.qrDataBus,normalize=normalize)
      self.apply_write()
      self.current_data_bus_dirty=True

    # def MeasureAll(self):
    #     self.qc.measure_all()

    def Measure(self):
        self.qc.add_register(self.crAddress)
        self.qc.add_register(self.crData)
        # self.qc.add_register(self.crReadFlag)

        self.qc.measure(self.qrAddressBus,self.crAddress)
        self.qc.measure(self.qrDataBus,self.crData)
        # self.qc.measure(self.qrReadFlagQubit,self.crReadFlag)

    def Measure_Internal_Data(self):
        self.qc.add_register(self.crAddress)
        self.qc.add_register(self.crData)
        # self.qc.add_register(self.crReadFlag)

        self.qc.measure(self.qrAddressRegister,self.crAddress)
        self.qc.measure(self.qrDataRegister,self.crData)
        # self.qc.measure(self.qrReadFlagQubit,self.crReadFlag)