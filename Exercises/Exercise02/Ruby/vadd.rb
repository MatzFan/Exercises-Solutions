#!/usr/bin/env ruby

require 'opencl_ruby_ffi'

SOURCE = <<EOF
__kernel void vector_add(__global float* A,
                         __global float* B,
                         __global float* C)
{
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}
EOF

a = [2, 4, 6, 5, 0, 8, 3]
b = [1, 1, 0, 5, 6, 3, 1]
h_a = NArray.sfloat(a.size).add! a
h_b = NArray.sfloat(b.size).add! b
h_c = NArray.sfloat(a.size) # filled with zeros

device = OpenCL::platforms.first.devices.last # GPU
context = OpenCL::create_context(device)
queue = context.create_command_queue device #, properties: OpenCL::CommandQueue::PROFILING_ENABLE
program = context.create_program_with_source(SOURCE).build

a_dev = context.create_buffer(h_a.size * h_a.element_size, flags: OpenCL::Mem::COPY_HOST_PTR, host_ptr: h_a)
b_dev = context.create_buffer(h_b.size * h_b.element_size, flags: OpenCL::Mem::COPY_HOST_PTR, host_ptr: h_b)
c_dev = context.create_buffer(h_c.size * h_c.element_size, flags: OpenCL::Mem::COPY_HOST_PTR, host_ptr: h_c)

event = program.vector_add(queue, h_a.shape, a_dev, b_dev, c_dev, local_work_size: [1])

queue.enqueue_read_buffer(c_dev, h_c, event_wait_list: [event])
queue.finish

puts h_c.to_a.inspect
