import argparse
import sys
import os
import subprocess
import tempfile

# This script simulates the Python/egglog front-end orchestrator.
# It generates a random quantum circuit of a given size, formats it as an egglog program,
# and then invokes the compiled Rust `egglog-experimental` CLI binary to run the program.

def generate_circuit_program(size: int, num_qubits: int = 20) -> str:
    import random
    
    # Generate random gates ahead of time
    gates = []
    gate_types = ["rust_apply_h_ffi", "rust_apply_t_ffi", "rust_apply_s_ffi", "rust_apply_z_ffi", "rust_apply_x_ffi"]
    
    for _ in range(size):
        gtype = random.choice(gate_types)
        q = random.randint(0, num_qubits - 1)
        gates.append((gtype, q))

    # Build the egglog script in chunks to prevent path variable overflow
    lines = []
    chunk_size = 30  # A safe depth to ensure we don't exceed the 41-variable limit
    num_chunks = (size + chunk_size - 1) // chunk_size
    gate_idx = 0

    for chunk_i in range(num_chunks):
        # Start each chunk with a fresh state to reset path variables
        # We add the '$' prefix to adhere to egglog strict mode convention
        # and prevent warnings from cluttering the terminal.
        lines.append(f"(let $state_chunk_{chunk_i}_0 (rust_id_pathsum_ffi {num_qubits}))")
        
        current_chunk_size = min(chunk_size, size - gate_idx)

        for i in range(current_chunk_size):
            gtype, q = gates[gate_idx]
            # Chain the 'let' statements within the chunk
            lines.append(f"(let $state_chunk_{chunk_i}_{i+1} ({gtype} $state_chunk_{chunk_i}_{i} {q}))")
            gate_idx += 1

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Egglog Full Stack Benchmark Script")
    parser.add_argument("--size", type=int, required=True, help="Number of gates to simulate")
    parser.add_argument("--bin", type=str, required=True, help="Path to the egglog-experimental binary")
    args = parser.parse_args()

    # 1. Generate the payload
    egglog_program = generate_circuit_program(args.size)
    
    # Write the payload to a system temporary file to avoid IDE file-watcher locks on Windows
    tmp_fd, tmp_file_path = tempfile.mkstemp(suffix=".egg", text=True)
    with os.fdopen(tmp_fd, 'w') as f:
        f.write(egglog_program)

    # 2. Execute the Rust binary, passing the .egg file
    # The egglog-experimental CLI automatically parses and runs files passed as arguments
    try:
        # The `capture_output` argument is removed to ensure that the output from the
        # Rust binary (including DHAT's summary) is printed directly to the console.
        result = subprocess.run([args.bin, tmp_file_path], text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running egglog binary. Exit code: {e.returncode}\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up the temp file from the system directory
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

if __name__ == "__main__":
    main()
