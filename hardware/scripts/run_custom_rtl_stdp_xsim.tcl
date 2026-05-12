#=============================================================================
# Run RTL-only STDP simulations with Vivado xsim.
#
# Usage from a Vivado command shell:
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_stdp_xsim.tcl
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_stdp_xsim.tcl -tclargs stdp
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_stdp_xsim.tcl -tclargs custom
#
# This script does not compile HLS RTL, does not instantiate design_1_wrapper,
# and does not require a board.
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

set rtl_dir    [file join $repo_root "hardware" "hdl" "rtl"]
set tb_dir     [file join $repo_root "hardware" "hdl" "tb"]
set inc_dir    [file join $repo_root "config" "generated"]
set work_dir   [file join $repo_root "hardware" "sim_work_rtl_stdp"]

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
set custom_top [file join $rtl_dir "top" "custom_rtl_stdp_sim_top.v"]

proc run_one {name top files inc_dir} {
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
    set sim_failed 0
    foreach line [split $data "\n"] {
        if {[regexp {(\[PASS\]|\[FAIL\]|Results:|PASSED|FAILED|ERROR|initial weight|updated weight|output spikes|STDP rule)} $line]} {
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

if {$selection eq "all" || $selection eq "stdp"} {
    run_one "tb_stdp_learning_engine" "tb_stdp_learning_engine" \
        [list $stdp_engine [file join $tb_dir "tb_stdp_learning_engine.v"]] \
        $inc_dir
}

if {$selection eq "all" || $selection eq "custom"} {
    run_one "tb_custom_rtl_stdp_learning" "tb_custom_rtl_stdp_learning" \
        [concat [list $stdp_engine] $rtl_core_files [list $custom_top [file join $tb_dir "tb_custom_rtl_stdp_learning.v"]]] \
        $inc_dir
}

puts ""
puts "Custom RTL STDP xsim run complete."
