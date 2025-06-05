from qiskit import *
from PIL import Image
import numpy as np

from qiskit.circuit.library import UCRYGate




def load_image(source_image,new_size=None):
  img_array=source_image
  if(isinstance( source_image , str)):
      img = Image.open(source_image).convert('L')
      if(new_size):
        img= img.resize(new_size)
      img_array = np.array(img)
  return img_array



def FRQI_encoding(qc,source_image,qr_position=None,qr_color=None,apply_h=True):
  n = int(np.ceil(np.log2(max(source_image.shape[:2]))))  # Number of qubits for positions
  if(qr_position is None):
    qr_position= QuantumRegister(2*n,name="Position")
    qc.add_register(qr_position)
  if(qr_color is None):
    qr_color= QuantumRegister(1,name="Color")
    qc.add_register(qr_color)

  qc_copy=qc.copy_empty_like(name=f"FRQI Image Encoding")
  if(apply_h):
    qc_copy.h(qr_position)

  flat_image = source_image.flatten()
  angles_list = list(2.0*np.arcsin(flat_image / 255.0))
  # print(source_image.shape)
  ucry= UCRYGate(angle_list=angles_list)
  qubits=[*qr_position]+[*qr_color]
  qc_copy.append(ucry,reversed(qubits))
  OFRQI_Gate= qc_copy.to_instruction()
  qc.append(OFRQI_Gate,qc.qubits)
  # qc.barrier()
  return qr_position,qr_color



def decode_frqi_image_aer(probabilities,n,use_zero_state=False):
  if(use_zero_state):
    filtered_dict = {k.replace(" ",""): v for k, v in probabilities.items() if k.startswith("0")}
  else:
    filtered_dict = {k.replace(" ",""): v for k, v in probabilities.items() if k.startswith("1")}

  restored_image= np.zeros((2**n,2**n))
  temp={}
  for k,v in filtered_dict.items():
    if(k in temp):
      print(f"repeated value:- key:{k} ,old:{temp[k]},new{v}")
    prob=0
    if(k not in temp or temp[k]<v):
      if(use_zero_state):
        prob= v-(1/(2**(2*n)))
      else:
        prob= v #if v>=0 else 0
      temp[k]=v
    xy_b=k[1:]
    x=xy_b[0:n][::-1]
    y=xy_b[n:2*n][::-1]
    val= int(255.0*np.sqrt(prob)*float(2**(2*n)))
    # print(v)
    # print(f"key:{k},x:{x},y:{y}")
    restored_image[int(y,2),int(x,2)]=val
  return restored_image

def group_dict_by_prefix(input_dict: dict, k: int) -> dict:
    """
    Splits a dictionary of bitstring-probability pairs into a larger dictionary
    where keys are prefixes and values are sub-dictionaries of matching entries.

    Args:
        input_dict (dict): The original dictionary where keys are bitstrings
                           and values are probabilities.
        k (int): The length of the bitstring prefix to group by.

    Returns:
        dict: A dictionary where:
              - Keys are the 'k'-bit prefixes.
              - Values are dictionaries containing the original bitstrings
                and their probabilities that start with that prefix.
    """
    grouped_data = {}
    for bitstring, probability in input_dict.items():
        if len(bitstring) < k:
            # Handle cases where a bitstring is shorter than the desired prefix length
            # You might want to raise an error, skip, or pad the bitstring
            print(f"Warning: Bitstring '{bitstring}' (length {len(bitstring)}) is shorter than prefix length {k}. Skipping.")
            continue

        prefix = bitstring[:k]
        suffix= bitstring[k:]
        # If the prefix is not yet a key in grouped_data, initialize it with an empty dictionary
        if prefix not in grouped_data:
            grouped_data[prefix] = {}

        # Add the original bitstring and its probability to the sub-dictionary
        grouped_data[prefix][suffix] = probability

    return grouped_data



  