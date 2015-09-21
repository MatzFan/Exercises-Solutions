#!/usr/bin/env ruby

require 'opencl_ruby_ffi'

p = OpenCL::platforms.first # Apple
devices = p.devices

puts "Platform: #{p.name}"
puts "Vendor: #{p.vendor}"
puts "Version: #{p.version}"

puts "Number of devices: #{devices.length}"

devices.each do |d|
  puts "\t-------------------------"
  puts "\t\tName: #{d.name}"
  puts "\t\tVersion: #{d.opencl_c_version}"
  puts "\t\tMax. Compute Units: #{d.max_compute_units}" # compute unit is a 'core' - broken into 'processing elements'
  puts "\t\tLocal Memory Size: #{d.local_mem_size/1024} KB"
  puts "\t\tGlobal Memory Size: #{d.global_mem_size/(1024*1024)} MB" # gloabl mem is RAM - or GPU equivalent
  puts "\t\tMax Alloc Size: #{d.max_mem_alloc_size/(1024*1024)} MB"
  puts "\t\tMax Work-group Total Size: #{d.max_work_group_size}"
  dim = d.max_work_item_sizes
  puts "\t\tMax Work-group Dims: #{dim.inspect}"
end
