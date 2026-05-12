#=============================================================================
# Run RTL-only unsupervised STDP simulations with Vivado xsim.
#
# Usage from a Vivado command shell:
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs unsup_fixed
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs mnist1
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs mnist10
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs batch
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs all
#
# This script is intentionally RTL-only: no HLS AXI wrapper, no snn_top_hls.v,
# and no design_1_wrapper.
#=============================================================================

set script_path [string map {\\ /} [info script]]
if {[file pathtype $script_path] ne "absolute"} {
    set script_path [file join [pwd] $script_path]
}
set script_dir [file dirname $script_path]
set repo_root  [file join $script_dir ".." ".."]

set candidate_roots [list]
if {[info exists env(USERPROFILE)]} {
    set desktop_root [file join [string map {\\ /} $env(USERPROFILE)] "Desktop" "Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA"]
    lappend candidate_roots $desktop_root
}
lappend candidate_roots "C:/Users/96898/Desktop/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA"
lappend candidate_roots $repo_root
lappend candidate_roots [pwd]

foreach candidate $candidate_roots {
    if {[file exists [file join $candidate "README.md"]]} {
        set repo_root $candidate
        break
    }
}

set rtl_dir  [file join $repo_root "hardware" "hdl" "rtl"]
set tb_dir   [file join $repo_root "hardware" "hdl" "tb"]
set inc_dir  [file join $repo_root "config" "generated"]
set work_dir [file join $repo_root "hardware" "sim_work_rtl_unsupervised"]
puts "Using repo root: $repo_root"
puts "Using work dir : $work_dir"

if {[llength $argv] > 0} {
    set selection [lindex $argv 0]
} else {
    set selection "all"
}

file mkdir $work_dir
cd $work_dir

set stdp_engine [file join $rtl_dir "learning" "stdp_learning_engine.v"]
set rtl_core_files [list \
    [file join $rtl_dir "core" "core_group.v"] \
    [file join $rtl_dir "core" "event_router_ng.v"] \
    [file join $rtl_dir "core" "synaptic_connectivity_table.v"] \
]
set unsup_top [file join $rtl_dir "top" "custom_rtl_unsupervised_sim_top.v"]
set data_dir  [file join $tb_dir "data"]

proc run_one {name top files inc_dir {spike_file ""}} {
    global work_dir

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

    set xvlog_cmd [list xvlog -sv -nolog -i $inc_dir]
    foreach f $files {
        lappend xvlog_cmd $f
    }

    puts "Compile: $compile_log"
    if {[catch {exec {*}$xvlog_cmd > $compile_log 2>@1} err]} {
        puts "COMPILE ERROR for $name"
        set fh [open $compile_log r]
        puts [read $fh]
        close $fh
        error $err
    }

    puts "Elaborate: $elab_log"
    if {[catch {exec xelab -nolog -debug typical $top -s ${name}_sim > $elab_log 2>@1} err]} {
        puts "ELABORATE ERROR for $name"
        set fh [open $elab_log r]
        puts [read $fh]
        close $fh
        error $err
    }

    set xsim_cmd [list xsim -nolog ${name}_sim]
    if {$spike_file ne ""} {
        if {![file exists $spike_file]} {
            error "Spike file not found: $spike_file. Run python software/python/generate_mnist_spike_files.py first."
        }
        file copy -force $spike_file [file join $work_dir "mnist_selected.mem"]
        lappend xsim_cmd -testplusarg "SPIKE_FILE=mnist_selected.mem"
    }
    lappend xsim_cmd -runall

    puts "Simulate: $sim_log"
    if {[catch {exec {*}$xsim_cmd > $sim_log 2>@1} err]} {
        puts "SIM ERROR for $name"
        set fh [open $sim_log r]
        puts [read $fh]
        close $fh
        error $err
    }

    set fh [open $sim_log r]
    set data [read $fh]
    close $fh
    set sim_failed 0
    foreach line [split $data "\n"] {
        if {[regexp {(\[PASS\]|\[FAIL\]|Results:|Summary:|PASSED|FAILED|ERROR|initial weights|natural output spike|updated weights|output spike|Unsupervised|winner|spike count|learned_updates)} $line]} {
            puts $line
        }
        if {[regexp {(\[FAIL\]|\[ERROR\]|FAILED)} $line]} {
            set sim_failed 1
        }
    }
    if {$sim_failed} {
        error "$name reported FAIL; see $sim_log"
    }
}

set unsup_fixed_files [concat \
    [list $stdp_engine] \
    $rtl_core_files \
    [list $unsup_top [file join $tb_dir "tb_custom_rtl_unsupervised_learning.v"]] \
]

if {$selection eq "all" || $selection eq "unsup_fixed"} {
    run_one "tb_custom_rtl_unsupervised_learning" "tb_custom_rtl_unsupervised_learning" \
        $unsup_fixed_files $inc_dir
}

set mnist_files [concat \
    [list $stdp_engine] \
    $rtl_core_files \
    [list $unsup_top [file join $tb_dir "tb_custom_rtl_unsupervised_mnist.v"]] \
]

if {$selection eq "all" || $selection eq "mnist1"} {
    run_one "tb_custom_rtl_unsupervised_mnist_mnist1" "tb_custom_rtl_unsupervised_mnist" \
        $mnist_files $inc_dir [file join $data_dir "mnist_unsup_1.mem"]
}

if {$selection eq "all" || $selection eq "mnist10"} {
    run_one "tb_custom_rtl_unsupervised_mnist_mnist10" "tb_custom_rtl_unsupervised_mnist" \
        $mnist_files $inc_dir [file join $data_dir "mnist_unsup_10.mem"]
}

if {$selection eq "all" || $selection eq "batch"} {
    run_one "tb_custom_rtl_unsupervised_mnist_batch" "tb_custom_rtl_unsupervised_mnist" \
        $mnist_files $inc_dir [file join $data_dir "mnist_unsup_batch.mem"]
}

puts ""
puts "Custom RTL unsupervised STDP xsim run complete."
