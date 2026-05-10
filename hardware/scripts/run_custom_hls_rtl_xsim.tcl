#=============================================================================
# Run custom generated-HLS/RTL simulations with Vivado xsim.
#
# Usage from a Vivado command shell:
#   vivado -mode batch -source hardware/scripts/run_custom_hls_rtl_xsim.tcl
#   vivado -mode batch -source hardware/scripts/run_custom_hls_rtl_xsim.tcl -tclargs hls
#   vivado -mode batch -source hardware/scripts/run_custom_hls_rtl_xsim.tcl -tclargs custom
#   vivado -mode batch -source hardware/scripts/run_custom_hls_rtl_xsim.tcl -tclargs mnist
#
# The script uses xvlog/xelab/xsim directly and does not require a board or
# design_1_wrapper.
#=============================================================================

set script_path [string map {\\ /} [info script]]
if {[file pathtype $script_path] ne "absolute"} {
    set script_path [file join [pwd] $script_path]
}
set script_dir [file dirname $script_path]
set repo_root  [file join $script_dir ".." ".."]

if {![file exists [file join $repo_root "README.md"]] && [file exists [file join [pwd] "README.md"]]} {
    set repo_root [pwd]
}
if {![file exists [file join $repo_root "README.md"]] && [info exists env(USERPROFILE)]} {
    set desktop_root [file join [string map {\\ /} $env(USERPROFILE)] "Desktop" "Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA"]
    if {[file exists [file join $desktop_root "README.md"]]} {
        set repo_root $desktop_root
    }
}

set rtl_dir    [file join $repo_root "hardware" "hdl" "rtl"]
set tb_dir     [file join $repo_root "hardware" "hdl" "tb"]
set hls_dir    [file join $repo_root "hardware" "hls" "hls_csim_output" "hls" "impl" "verilog"]
set inc_dir    [file join $repo_root "config" "generated"]
set work_dir   [file join $repo_root "hardware" "sim_work_custom"]

if {[llength $argv] > 0} {
    set selection [lindex $argv 0]
} else {
    set selection "all"
}

file mkdir $work_dir
cd $work_dir

foreach dat [glob -nocomplain [file join $hls_dir "*.dat"]] {
    file copy -force $dat [file join $work_dir [file tail $dat]]
}
foreach mem [glob -nocomplain [file join $tb_dir "data" "*.mem"]] {
    file copy -force $mem [file join $work_dir [file tail $mem]]
}

set hls_files [lsort [glob -nocomplain [file join $hls_dir "*.v"]]]
set rtl_core_files [list \
    [file join $rtl_dir "core" "core_group.v"] \
    [file join $rtl_dir "core" "event_router_ng.v"] \
    [file join $rtl_dir "core" "synaptic_connectivity_table.v"] \
]
set custom_top [file join $rtl_dir "top" "custom_hls_rtl_sim_top.v"]

proc run_one {name top files inc_dir} {
    global hls_dir
    puts ""
    puts "============================================================"
    puts "Running $name"
    puts "============================================================"

    set compile_log "compile_${name}.log"
    set elab_log "elab_${name}.log"
    set sim_log "sim_${name}.log"

    catch {file delete -force xsim.dir}
    foreach f [list $compile_log $elab_log $sim_log] {
        catch {file delete -force $f}
    }
    foreach f [glob -nocomplain *.jou *.pb *.wdb] {
        catch {file delete -force $f}
    }

    set xvlog_cmd [list xvlog -nolog -i $inc_dir -i $hls_dir]
    foreach f $files {
        lappend xvlog_cmd $f
    }
    puts "Compile: $compile_log"
    if {[catch {exec {*}$xvlog_cmd > $compile_log 2>@1} err]} {
        puts "COMPILE ERROR for $name"
        puts [read [open $compile_log r]]
        error $err
    }

    puts "Elaborate: $elab_log"
    if {[catch {exec xelab -nolog -debug typical $top -s ${name}_sim > $elab_log 2>@1} err]} {
        puts "ELABORATE ERROR for $name"
        puts [read [open $elab_log r]]
        error $err
    }

    puts "Simulate: $sim_log"
    if {[catch {exec xsim -nolog ${name}_sim -runall > $sim_log 2>@1} err]} {
        puts "SIM ERROR for $name"
        puts [read [open $sim_log r]]
        error $err
    }

    set fh [open $sim_log r]
    set data [read $fh]
    close $fh
    foreach line [split $data "\n"] {
        if {[regexp {(\[PASS\]|\[FAIL\]|Results:|PASSED|FAILED|ERROR|initial weight|pre neuron|output spike)} $line]} {
            puts $line
        }
    }
}

if {$selection eq "all" || $selection eq "hls"} {
    run_one "tb_hls_learning_engine" "tb_hls_learning_engine" \
        [concat $hls_files [list [file join $tb_dir "tb_hls_learning_engine.v"]]] \
        $inc_dir
}

if {$selection eq "all" || $selection eq "custom"} {
    run_one "tb_custom_hls_rtl_learning" "tb_custom_hls_rtl_learning" \
        [concat $hls_files $rtl_core_files [list $custom_top [file join $tb_dir "tb_custom_hls_rtl_learning.v"]]] \
        $inc_dir
}

if {$selection eq "all" || $selection eq "mnist"} {
    run_one "tb_mnist_hls_rtl_sim" "tb_mnist_hls_rtl_sim" \
        [concat $hls_files $rtl_core_files [list $custom_top [file join $tb_dir "tb_mnist_hls_rtl_sim.v"]]] \
        $inc_dir
}

puts ""
puts "Custom HLS/RTL xsim run complete."
