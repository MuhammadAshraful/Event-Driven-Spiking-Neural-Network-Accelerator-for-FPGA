//=============================================================================
// Two-core_group + CT MNIST classifier testbench
//
// Training is unsupervised. Labels are used only for output-neuron assignment
// and final evaluation after learning is disabled.
//=============================================================================

`timescale 1ns / 1ps

module tb_custom_rtl_mnist_twogroup_ct_classifier;

    localparam INPUT_ID_WIDTH  = 6;
    localparam OUTPUT_ID_WIDTH = 4;
    localparam MAX_EVENTS      = 30000;
    localparam MAX_IMAGES      = 1200;

    reg clk;
    reg rst_n;
    reg enable;
    reg learning_enable;
    reg image_start;
    reg event_valid;
    reg [INPUT_ID_WIDTH-1:0] event_neuron_id;
    reg [3:0] event_weight;
    reg image_end;

    wire event_ready;
    wire init_done;
    wire image_done;
    wire winner_valid;
    wire [OUTPUT_ID_WIDTH-1:0] winner_id;
    wire [31:0] latency_cycles;
    wire [15:0] image_output_spike_count;
    wire [31:0] total_images;
    wire [31:0] total_input_spikes;
    wire [31:0] total_group0_spikes;
    wire [31:0] total_group1_spikes;
    wire [31:0] total_ct_changed_weights;
    wire [7:0] debug_weight_min;
    wire [7:0] debug_weight_max;
    wire [31:0] debug_weight_sum;
    wire [159:0] debug_train_win_counts;
    wire [15:0] debug_dead_outputs;
    wire [15:0] debug_dominant_wins;
    wire [31:0] router_routed_spike_count;
    wire [31:0] router_observed_spike_count;
    wire [31:0] inter_group_routed_spikes;
    wire [31:0] ct_init_write_count;
    wire [31:0] ct_learned_update_count;
    wire [31:0] direct_learning_write_count;
    wire [31:0] direct_ct_write_count;

    integer train_cycle [0:MAX_EVENTS-1];
    integer train_neuron [0:MAX_EVENTS-1];
    integer train_weight [0:MAX_EVENTS-1];
    integer train_image [0:MAX_EVENTS-1];
    integer train_phase [0:MAX_EVENTS-1];
    integer test_cycle [0:MAX_EVENTS-1];
    integer test_neuron [0:MAX_EVENTS-1];
    integer test_weight [0:MAX_EVENTS-1];
    integer test_image [0:MAX_EVENTS-1];
    integer test_phase [0:MAX_EVENTS-1];
    integer train_labels [0:MAX_IMAGES-1];
    integer test_labels [0:MAX_IMAGES-1];
    integer assign_counts [0:9][0:9];
    integer assigned_labels [0:9];
    integer confusion [0:9][0:9];
    integer output_win_counts [0:9];
    integer train_output_win_counts [0:9];
    integer train_event_count;
    integer test_event_count;
    integer train_image_count;
    integer test_image_count;
    integer correct_count;
    integer latency_sum;
    integer output_spike_sum;
    integer accuracy_permille;
    integer pass_count;
    integer fail_count;
    integer sim_cycle;
    string subset_name;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (!rst_n)
            sim_cycle <= 0;
        else
            sim_cycle <= sim_cycle + 1;
    end

    custom_rtl_mnist_twogroup_ct_classifier_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .learning_enable(learning_enable),
        .image_start(image_start),
        .event_valid(event_valid),
        .event_neuron_id(event_neuron_id),
        .event_weight(event_weight),
        .image_end(image_end),
        .event_ready(event_ready),
        .init_done(init_done),
        .image_done(image_done),
        .winner_valid(winner_valid),
        .winner_id(winner_id),
        .latency_cycles(latency_cycles),
        .image_output_spike_count(image_output_spike_count),
        .total_images(total_images),
        .total_input_spikes(total_input_spikes),
        .total_group0_spikes(total_group0_spikes),
        .total_group1_spikes(total_group1_spikes),
        .total_ct_changed_weights(total_ct_changed_weights),
        .debug_weight_min(debug_weight_min),
        .debug_weight_max(debug_weight_max),
        .debug_weight_sum(debug_weight_sum),
        .debug_train_win_counts(debug_train_win_counts),
        .debug_dead_outputs(debug_dead_outputs),
        .debug_dominant_wins(debug_dominant_wins),
        .router_routed_spike_count(router_routed_spike_count),
        .router_observed_spike_count(router_observed_spike_count),
        .inter_group_routed_spikes(inter_group_routed_spikes),
        .ct_init_write_count(ct_init_write_count),
        .ct_learned_update_count(ct_learned_update_count),
        .direct_learning_write_count(direct_learning_write_count),
        .direct_ct_write_count(direct_ct_write_count)
    );

    task automatic check;
        input [8*128-1:0] desc;
        input condition;
    begin
        if (condition) begin
            $display("[PASS] %0s", desc);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] %0s", desc);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task automatic do_reset;
        integer timeout;
    begin
        rst_n <= 1'b0;
        enable <= 1'b0;
        learning_enable <= 1'b0;
        image_start <= 1'b0;
        event_valid <= 1'b0;
        event_neuron_id <= 6'd0;
        event_weight <= 4'd0;
        image_end <= 1'b0;
        repeat (30) @(posedge clk);
        rst_n <= 1'b1;
        enable <= 1'b1;

        timeout = 0;
        while (!init_done && timeout < 30000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (!init_done) begin
            $display("[ERROR] two-group CT classifier init timed out");
            $finish(2);
        end
        repeat (20) @(posedge clk);
    end
    endtask

    task automatic load_events;
        input string filename;
        output integer event_count;
        output integer image_count;
        output integer cycles [0:MAX_EVENTS-1];
        output integer neurons [0:MAX_EVENTS-1];
        output integer weights [0:MAX_EVENTS-1];
        output integer images [0:MAX_EVENTS-1];
        output integer phases [0:MAX_EVENTS-1];
        integer fh;
        integer code;
        integer scratch;
    begin
        fh = $fopen(filename, "r");
        if (fh == 0) begin
            $display("[ERROR] could not open event file %0s", filename);
            $finish(2);
        end

        event_count = 0;
        image_count = 0;
        while (!$feof(fh) && event_count < MAX_EVENTS) begin
            code = $fscanf(fh, "%d %d %d %d %d\n",
                           cycles[event_count],
                           neurons[event_count],
                           weights[event_count],
                           images[event_count],
                           phases[event_count]);
            if (code == 5) begin
                if (images[event_count] + 1 > image_count)
                    image_count = images[event_count] + 1;
                event_count = event_count + 1;
            end else begin
                scratch = $fgetc(fh);
            end
        end
        $fclose(fh);
        $display("loaded events: %0s images=%0d events=%0d", filename, image_count, event_count);
    end
    endtask

    task automatic load_labels;
        input string filename;
        output integer image_count;
        output integer labels [0:MAX_IMAGES-1];
        integer fh;
        integer code;
        integer image_id;
        integer label;
        string header;
    begin
        fh = $fopen(filename, "r");
        if (fh == 0) begin
            $display("[ERROR] could not open label file %0s", filename);
            $finish(2);
        end
        image_count = 0;
        void'($fgets(header, fh));
        while (!$feof(fh) && image_count < MAX_IMAGES) begin
            code = $fscanf(fh, "%d %d\n", image_id, label);
            if (code == 2) begin
                labels[image_id] = label;
                if (image_id + 1 > image_count)
                    image_count = image_id + 1;
            end
        end
        $fclose(fh);
        $display("loaded labels: %0s images=%0d", filename, image_count);
    end
    endtask

    task automatic send_event;
        input integer neuron_id;
        input integer weight;
        integer timeout;
    begin
        @(negedge clk);
        event_neuron_id <= neuron_id[INPUT_ID_WIDTH-1:0];
        event_weight <= weight[3:0];
        event_valid <= 1'b1;

        timeout = 0;
        while (!event_ready && timeout < 250000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (!event_ready) begin
            $display("[ERROR] event_ready timeout neuron=%0d", neuron_id);
            $finish(3);
        end

        @(posedge clk);
        @(negedge clk);
        event_valid <= 1'b0;
        event_weight <= 4'd0;
        event_neuron_id <= 6'd0;
    end
    endtask

    task automatic run_image;
        input integer image_id;
        input integer event_count;
        input integer cycles [0:MAX_EVENTS-1];
        input integer neurons [0:MAX_EVENTS-1];
        input integer weights [0:MAX_EVENTS-1];
        input integer images [0:MAX_EVENTS-1];
        output integer winner;
        output integer had_winner;
        output integer latency;
        output integer out_spikes;
        integer idx;
        integer timeout;
    begin
        @(posedge clk);
        image_start <= 1'b1;
        @(posedge clk);
        image_start <= 1'b0;

        for (idx = 0; idx < event_count; idx = idx + 1) begin
            if (images[idx] == image_id)
                send_event(neurons[idx], weights[idx]);
        end

        @(posedge clk);
        image_end <= 1'b1;
        @(posedge clk);
        image_end <= 1'b0;

        timeout = 0;
        while (!image_done && timeout < 500000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (!image_done) begin
            $display("[ERROR] image_done timeout image=%0d", image_id);
            $finish(4);
        end

        had_winner = winner_valid;
        winner = winner_valid ? winner_id : 0;
        latency = latency_cycles;
        out_spikes = image_output_spike_count;
        repeat (12) @(posedge clk);
    end
    endtask

    task automatic run_training_pass;
        integer img;
        integer winner;
        integer had_winner;
        integer latency;
        integer out_spikes;
        integer out_idx;
    begin
        for (out_idx = 0; out_idx < 10; out_idx = out_idx + 1)
            train_output_win_counts[out_idx] = 0;
        learning_enable <= 1'b1;
        repeat (5) @(posedge clk);
        for (img = 0; img < train_image_count; img = img + 1) begin
            run_image(img, train_event_count, train_cycle, train_neuron, train_weight, train_image,
                      winner, had_winner, latency, out_spikes);
            if (had_winner)
                train_output_win_counts[winner] = train_output_win_counts[winner] + 1;
            $display("twogroup train image=%0d winner=%0d valid=%0d out_spikes=%0d latency=%0d",
                     img, winner, had_winner, out_spikes, latency);
        end
        learning_enable <= 1'b0;
    end
    endtask

    task automatic run_assignment_pass;
        integer img;
        integer out_idx;
        integer digit;
        integer winner;
        integer had_winner;
        integer latency;
        integer out_spikes;
        integer best_digit;
        integer best_count;
    begin
        for (out_idx = 0; out_idx < 10; out_idx = out_idx + 1) begin
            assigned_labels[out_idx] = 0;
            output_win_counts[out_idx] = 0;
            for (digit = 0; digit < 10; digit = digit + 1)
                assign_counts[out_idx][digit] = 0;
        end

        learning_enable <= 1'b0;
        repeat (5) @(posedge clk);
        for (img = 0; img < train_image_count; img = img + 1) begin
            run_image(img, train_event_count, train_cycle, train_neuron, train_weight, train_image,
                      winner, had_winner, latency, out_spikes);
            if (had_winner) begin
                assign_counts[winner][train_labels[img]] = assign_counts[winner][train_labels[img]] + 1;
                output_win_counts[winner] = output_win_counts[winner] + 1;
            end
        end

        for (out_idx = 0; out_idx < 10; out_idx = out_idx + 1) begin
            best_digit = 0;
            best_count = -1;
            for (digit = 0; digit < 10; digit = digit + 1) begin
                if (assign_counts[out_idx][digit] > best_count) begin
                    best_count = assign_counts[out_idx][digit];
                    best_digit = digit;
                end
            end
            assigned_labels[out_idx] = best_digit;
            $display("twogroup assigned output %0d -> digit %0d count=%0d wins=%0d",
                     out_idx, best_digit, best_count, output_win_counts[out_idx]);
        end
    end
    endtask

    task automatic run_test_pass;
        integer img;
        integer digit;
        integer pred;
        integer winner;
        integer had_winner;
        integer latency;
        integer out_spikes;
    begin
        correct_count = 0;
        latency_sum = 0;
        output_spike_sum = 0;
        for (digit = 0; digit < 10; digit = digit + 1) begin
            for (pred = 0; pred < 10; pred = pred + 1)
                confusion[digit][pred] = 0;
        end

        learning_enable <= 1'b0;
        repeat (5) @(posedge clk);
        for (img = 0; img < test_image_count; img = img + 1) begin
            run_image(img, test_event_count, test_cycle, test_neuron, test_weight, test_image,
                      winner, had_winner, latency, out_spikes);
            pred = had_winner ? assigned_labels[winner] : 0;
            digit = test_labels[img];
            confusion[digit][pred] = confusion[digit][pred] + 1;
            if (pred == digit)
                correct_count = correct_count + 1;
            latency_sum = latency_sum + latency;
            output_spike_sum = output_spike_sum + out_spikes;
            $display("twogroup test image=%0d label=%0d winner=%0d valid=%0d pred=%0d out_spikes=%0d latency=%0d",
                     img, digit, winner, had_winner, pred, out_spikes, latency);
        end
    end
    endtask

    initial begin
        string train_file;
        string test_file;
        string train_label_file;
        string test_label_file;
        integer train_label_count;
        integer test_label_count;
        integer avg_latency;
        integer avg_output_x1000;
        integer avg_input_x1000;
        integer avg_group0_x1000;
        integer avg_group1_x1000;
        integer throughput_ips;
        integer dead_outputs;
        integer dominant_wins;
        integer dominant_output;
        integer dominant_ratio_permille;
        integer train_dead_outputs;
        integer train_dominant_wins;

        pass_count = 0;
        fail_count = 0;

        if (!$value$plusargs("SUBSET=%s", subset_name))
            subset_name = "10";
        if (!$value$plusargs("TRAIN_FILE=%s", train_file))
            train_file = "classifier_train.mem";
        if (!$value$plusargs("TEST_FILE=%s", test_file))
            test_file = "classifier_test.mem";
        if (!$value$plusargs("TRAIN_LABEL_FILE=%s", train_label_file))
            train_label_file = "classifier_train_labels.txt";
        if (!$value$plusargs("TEST_LABEL_FILE=%s", test_label_file))
            test_label_file = "classifier_test_labels.txt";

        $display("=========================================================");
        $display("  Two-group CT MNIST classifier TB subset=%0s", subset_name);
        $display("=========================================================");

        load_events(train_file, train_event_count, train_image_count,
                    train_cycle, train_neuron, train_weight, train_image, train_phase);
        load_events(test_file, test_event_count, test_image_count,
                    test_cycle, test_neuron, test_weight, test_image, test_phase);
        load_labels(train_label_file, train_label_count, train_labels);
        load_labels(test_label_file, test_label_count, test_labels);

        do_reset;
        run_training_pass;
        run_assignment_pass;
        run_test_pass;

        accuracy_permille = (correct_count * 1000) / test_image_count;
        avg_latency = latency_sum / test_image_count;
        avg_output_x1000 = (output_spike_sum * 1000) / test_image_count;
        avg_input_x1000 = (test_event_count * 1000) / test_image_count;
        avg_group0_x1000 = (total_group0_spikes * 1000) / total_images;
        avg_group1_x1000 = (total_group1_spikes * 1000) / total_images;
        throughput_ips = 100000000 / (avg_latency + 1);
        dead_outputs = 0;
        dominant_wins = 0;
        dominant_output = 0;
        for (integer usage_idx = 0; usage_idx < 10; usage_idx = usage_idx + 1) begin
            if (output_win_counts[usage_idx] == 0)
                dead_outputs = dead_outputs + 1;
            if (output_win_counts[usage_idx] > dominant_wins) begin
                dominant_wins = output_win_counts[usage_idx];
                dominant_output = usage_idx;
            end
        end
        dominant_ratio_permille = (dominant_wins * 1000) / train_image_count;
        train_dead_outputs = 0;
        train_dominant_wins = 0;
        for (integer train_usage_idx = 0; train_usage_idx < 10; train_usage_idx = train_usage_idx + 1) begin
            if (train_output_win_counts[train_usage_idx] == 0)
                train_dead_outputs = train_dead_outputs + 1;
            if (train_output_win_counts[train_usage_idx] > train_dominant_wins)
                train_dominant_wins = train_output_win_counts[train_usage_idx];
        end

        $display("twogroup confusion matrix:");
        for (integer r = 0; r < 10; r = r + 1) begin
            $display("%0d: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                     r,
                     confusion[r][0], confusion[r][1], confusion[r][2], confusion[r][3], confusion[r][4],
                     confusion[r][5], confusion[r][6], confusion[r][7], confusion[r][8], confusion[r][9]);
        end
        $display("TWOGROUP_CT_ASSIGN_WIN_COUNTS: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 output_win_counts[0], output_win_counts[1], output_win_counts[2], output_win_counts[3], output_win_counts[4],
                 output_win_counts[5], output_win_counts[6], output_win_counts[7], output_win_counts[8], output_win_counts[9]);
        $display("TWOGROUP_CT_ASSIGNED_LABELS: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 assigned_labels[0], assigned_labels[1], assigned_labels[2], assigned_labels[3], assigned_labels[4],
                 assigned_labels[5], assigned_labels[6], assigned_labels[7], assigned_labels[8], assigned_labels[9]);
        $display("TWOGROUP_CT_TRAIN_WIN_COUNTS: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 train_output_win_counts[0], train_output_win_counts[1], train_output_win_counts[2], train_output_win_counts[3], train_output_win_counts[4],
                 train_output_win_counts[5], train_output_win_counts[6], train_output_win_counts[7], train_output_win_counts[8], train_output_win_counts[9]);
        $display("TWOGROUP_CT_GROUP1_INPUT_COUNTS: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 dut.group1_input_count[0], dut.group1_input_count[1], dut.group1_input_count[2], dut.group1_input_count[3], dut.group1_input_count[4],
                 dut.group1_input_count[5], dut.group1_input_count[6], dut.group1_input_count[7], dut.group1_input_count[8], dut.group1_input_count[9]);
        $display("TWOGROUP_CT_GROUP1_OUTPUT_COUNTS: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 dut.group1_output_count[0], dut.group1_output_count[1], dut.group1_output_count[2], dut.group1_output_count[3], dut.group1_output_count[4],
                 dut.group1_output_count[5], dut.group1_output_count[6], dut.group1_output_count[7], dut.group1_output_count[8], dut.group1_output_count[9]);

        check("CT initialized 64x10 inter-group entries through router path", ct_init_write_count == 640);
        check("group 0 input neurons fired naturally", total_group0_spikes > 0);
        check("event_router_ng and CT routed inter-group spikes to group 1", inter_group_routed_spikes > 0);
        check("group 1 output neurons fired naturally", output_spike_sum > 0);
        check("router learn_spike observation path saw group spikes", router_observed_spike_count > 0);
        check("learned inter-group updates used event_router_ng.learn_weight_*", ct_learned_update_count > 0);
        check("direct CT/core_group learning writes avoided", direct_ct_write_count == 0 && direct_learning_write_count == 0);
        check("assignment and test images processed", test_image_count > 0 && train_image_count > 0);

        $display("MNIST_TWOGROUP_CT_RTL_SUMMARY subset=%0s train_images=%0d test_images=%0d accuracy_permille=%0d avg_latency_cycles=%0d avg_output_spikes_x1000=%0d avg_input_spikes_x1000=%0d avg_group0_spikes_x1000=%0d avg_group1_spikes_x1000=%0d throughput_images_per_sec=%0d ct_entries=640 weight_min=%0d weight_max=%0d weight_sum=%0d ct_changed_weights=%0d ct_learned_updates=%0d ct_init_writes=%0d router_routed_spikes=%0d router_observed_spikes=%0d inter_group_routed_spikes=%0d direct_learning_writes=%0d direct_ct_writes=%0d dead_outputs=%0d dominant_output=%0d dominant_wins=%0d dominant_ratio_permille=%0d train_dead_outputs=%0d train_dominant_wins=%0d sim_cycles=%0d",
                 subset_name, train_image_count, test_image_count, accuracy_permille,
                 avg_latency, avg_output_x1000, avg_input_x1000, avg_group0_x1000,
                 avg_group1_x1000, throughput_ips, debug_weight_min, debug_weight_max,
                 debug_weight_sum, total_ct_changed_weights, ct_learned_update_count,
                 ct_init_write_count, router_routed_spike_count, router_observed_spike_count,
                 inter_group_routed_spikes, direct_learning_write_count, direct_ct_write_count,
                 dead_outputs, dominant_output, dominant_wins, dominant_ratio_permille,
                 train_dead_outputs, train_dominant_wins, sim_cycle);

        if (fail_count != 0) begin
            $display("*** MNIST TWOGROUP CT CLASSIFIER RTL TEST FAILED ***");
            $finish(1);
        end

        $display("*** MNIST TWOGROUP CT CLASSIFIER RTL TEST PASSED ***");
        $finish(0);
    end

    initial begin
        #400000000;
        $display("[ERROR] two-group CT classifier TB timed out");
        $finish(2);
    end

endmodule
