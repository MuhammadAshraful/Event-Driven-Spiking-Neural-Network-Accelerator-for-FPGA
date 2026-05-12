#=============================================================================
# Run one-core_group MNIST classifier simulations with Vivado xsim.
#
# Usage:
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_coregroup_classifier_xsim.tcl -tclargs mnist10
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_coregroup_classifier_xsim.tcl -tclargs mnist100
#   vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_coregroup_classifier_xsim.tcl -tclargs all
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
set data_dir [file join $tb_dir "data"]
set inc_dir  [file join $repo_root "config" "generated"]
set work_dir [file join $repo_root "hardware" "sim_work_rtl_mnist_coregroup_classifier"]

if {[llength $argv] > 0} {
    set selection [lindex $argv 0]
} else {
    set selection "all"
}

file mkdir $work_dir
cd $work_dir

set files [list \
    [file join $rtl_dir "learning" "winner_take_all.v"] \
    [file join $rtl_dir "core" "core_group.v"] \
    [file join $rtl_dir "top" "custom_rtl_mnist_coregroup_classifier_top.v"] \
    [file join $tb_dir "tb_custom_rtl_mnist_coregroup_classifier.v"] \
]

proc run_one {subset files inc_dir data_dir work_dir} {
    set name "tb_custom_rtl_mnist_coregroup_classifier_${subset}"
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

    set train_file [file join $data_dir "mnist_classifier_train_${subset}.mem"]
    set test_file [file join $data_dir "mnist_classifier_test_${subset}.mem"]
    set train_labels [file join $data_dir "mnist_classifier_train_${subset}_labels.txt"]
    set test_labels [file join $data_dir "mnist_classifier_test_${subset}_labels.txt"]

    foreach f [list $train_file $test_file $train_labels $test_labels] {
        if {![file exists $f]} {
            error "Missing $f. Run python software/python/generate_mnist_classifier_files.py first."
        }
    }

    file copy -force $train_file [file join $work_dir "classifier_train.mem"]
    file copy -force $test_file [file join $work_dir "classifier_test.mem"]
    file copy -force $train_labels [file join $work_dir "classifier_train_labels.txt"]
    file copy -force $test_labels [file join $work_dir "classifier_test_labels.txt"]

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
    if {[catch {exec xelab -nolog -debug typical tb_custom_rtl_mnist_coregroup_classifier -s ${name}_sim > $elab_log 2>@1} err]} {
        puts "ELABORATE ERROR for $name"
        set fh [open $elab_log r]
        puts [read $fh]
        close $fh
        error $err
    }

    set xsim_cmd [list xsim -nolog ${name}_sim \
        -testplusarg "SUBSET=${subset}" \
        -testplusarg "TRAIN_FILE=classifier_train.mem" \
        -testplusarg "TEST_FILE=classifier_test.mem" \
        -testplusarg "TRAIN_LABEL_FILE=classifier_train_labels.txt" \
        -testplusarg "TEST_LABEL_FILE=classifier_test_labels.txt" \
        -runall]

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
        if {[regexp {(\[PASS\]|\[FAIL\]|ERROR|MNIST_COREGROUP_RTL_SUMMARY|assigned output|COREGROUP CLASSIFIER)} $line]} {
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

if {$selection eq "all" || $selection eq "mnist10"} {
    run_one "10" $files $inc_dir $data_dir $work_dir
}

if {$selection eq "all" || $selection eq "mnist100"} {
    run_one "100" $files $inc_dir $data_dir $work_dir
}

if {$selection eq "mnist1000"} {
    run_one "1000" $files $inc_dir $data_dir $work_dir
}

puts ""
puts "Custom RTL MNIST coregroup classifier xsim run complete."
