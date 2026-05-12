//=============================================================================
// RTL MNIST classifier testbench
//
// Flow:
//   1. Train with learning enabled. No labels are used.
//   2. Replay the training set with learning disabled and use labels only to
//      assign output neurons to digit classes.
//   3. Replay the test set with learning disabled and report classifier metrics.
//=============================================================================

`timescale 1ns / 1ps

module tb_custom_rtl_mnist_classifier;

    localparam INPUT_ID_WIDTH = 6;
    localparam OUTPUT_ID_WIDTH = 4;
    localparam MAX_EVENTS = 30000;
    localparam MAX_IMAGES = 1200;

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
    wire image_done;
    wire winner_valid;
    wire [OUTPUT_ID_WIDTH-1:0] winner_id;
    wire [31:0] latency_cycles;
    wire [31:0] total_images;
    wire [31:0] total_output_spikes;
    wire [31:0] total_weight_updates;
    wire [7:0] debug_weight_min;
    wire [7:0] debug_weight_max;
    wire [31:0] debug_weight_sum;

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
    integer train_event_count;
    integer test_event_count;
    integer train_image_count;
    integer test_image_count;
    integer correct_count;
    integer latency_sum;
    integer output_spike_count;
    integer accuracy_permille;
    string subset_name;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    custom_rtl_mnist_classifier_top dut (
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
        .image_done(image_done),
        .winner_valid(winner_valid),
        .winner_id(winner_id),
        .latency_cycles(latency_cycles),
        .total_images(total_images),
        .total_output_spikes(total_output_spikes),
        .total_weight_updates(total_weight_updates),
        .debug_weight_min(debug_weight_min),
        .debug_weight_max(debug_weight_max),
        .debug_weight_sum(debug_weight_sum)
    );

    task automatic do_reset;
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
        repeat (50) @(posedge clk);
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
        header = "";
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
        while (!event_ready && timeout < 100000) begin
            @(posedge clk);
            timeout = timeout + 1;
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
        output integer latency;
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
        while (!image_done && timeout < 100000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        if (timeout >= 100000) begin
            $display("[ERROR] timeout waiting for image_done image=%0d", image_id);
            $finish(3);
        end

        winner = winner_id;
        latency = latency_cycles;
        repeat (12) @(posedge clk);
    end
    endtask

    task automatic run_training_pass;
        integer img;
        integer winner;
        integer latency;
    begin
        learning_enable <= 1'b1;
        repeat (5) @(posedge clk);
        for (img = 0; img < train_image_count; img = img + 1) begin
            run_image(img, train_event_count, train_cycle, train_neuron, train_weight, train_image, winner, latency);
            if ((img % 25) == 0)
                $display("train image=%0d winner=%0d latency=%0d", img, winner, latency);
        end
        learning_enable <= 1'b0;
    end
    endtask

    task automatic run_assignment_pass;
        integer img;
        integer out_idx;
        integer digit;
        integer winner;
        integer latency;
        integer best_digit;
        integer best_count;
    begin
        for (out_idx = 0; out_idx < 10; out_idx = out_idx + 1) begin
            assigned_labels[out_idx] = 0;
            for (digit = 0; digit < 10; digit = digit + 1)
                assign_counts[out_idx][digit] = 0;
        end

        learning_enable <= 1'b0;
        repeat (5) @(posedge clk);
        for (img = 0; img < train_image_count; img = img + 1) begin
            run_image(img, train_event_count, train_cycle, train_neuron, train_weight, train_image, winner, latency);
            assign_counts[winner][train_labels[img]] = assign_counts[winner][train_labels[img]] + 1;
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
            $display("assigned output %0d -> digit %0d count=%0d", out_idx, best_digit, best_count);
        end
    end
    endtask

    task automatic run_test_pass;
        integer img;
        integer digit;
        integer pred;
        integer winner;
        integer latency;
    begin
        correct_count = 0;
        latency_sum = 0;
        output_spike_count = 0;
        for (digit = 0; digit < 10; digit = digit + 1) begin
            for (pred = 0; pred < 10; pred = pred + 1)
                confusion[digit][pred] = 0;
        end

        learning_enable <= 1'b0;
        repeat (5) @(posedge clk);
        for (img = 0; img < test_image_count; img = img + 1) begin
            run_image(img, test_event_count, test_cycle, test_neuron, test_weight, test_image, winner, latency);
            pred = assigned_labels[winner];
            digit = test_labels[img];
            confusion[digit][pred] = confusion[digit][pred] + 1;
            if (pred == digit)
                correct_count = correct_count + 1;
            latency_sum = latency_sum + latency;
            output_spike_count = output_spike_count + 1;
            $display("test image=%0d label=%0d winner=%0d pred=%0d latency=%0d",
                     img, digit, winner, pred, latency);
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

        if (!$value$plusargs("SUBSET=%s", subset_name))
            subset_name = "mnist10";
        if (!$value$plusargs("TRAIN_FILE=%s", train_file))
            train_file = "classifier_train.mem";
        if (!$value$plusargs("TEST_FILE=%s", test_file))
            test_file = "classifier_test.mem";
        if (!$value$plusargs("TRAIN_LABEL_FILE=%s", train_label_file))
            train_label_file = "classifier_train_labels.txt";
        if (!$value$plusargs("TEST_LABEL_FILE=%s", test_label_file))
            test_label_file = "classifier_test_labels.txt";

        $display("=========================================================");
        $display("  RTL MNIST classifier TB subset=%0s", subset_name);
        $display("=========================================================");

        load_events(train_file, train_event_count, train_image_count,
                    train_cycle, train_neuron, train_weight, train_image, train_phase);
        load_events(test_file, test_event_count, test_image_count,
                    test_cycle, test_neuron, test_weight, test_image, test_phase);
        load_labels(train_label_file, train_label_count, train_labels);
        load_labels(test_label_file, test_label_count, test_labels);

        if (train_label_count < train_image_count || test_label_count < test_image_count) begin
            $display("[ERROR] label count does not cover event images");
            $finish(4);
        end

        do_reset;
        run_training_pass;
        run_assignment_pass;
        run_test_pass;

        accuracy_permille = (correct_count * 1000) / test_image_count;
        avg_latency = latency_sum / test_image_count;
        avg_output_x1000 = (output_spike_count * 1000) / test_image_count;

        $display("confusion matrix:");
        for (integer r = 0; r < 10; r = r + 1) begin
            $display("%0d: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                     r,
                     confusion[r][0], confusion[r][1], confusion[r][2], confusion[r][3], confusion[r][4],
                     confusion[r][5], confusion[r][6], confusion[r][7], confusion[r][8], confusion[r][9]);
        end

        $display("MNIST_CLASSIFIER_RTL_SUMMARY subset=%0s train_images=%0d test_images=%0d assigned_accuracy_permille=%0d avg_latency_cycles=%0d avg_output_spikes_x1000=%0d weight_min=%0d weight_max=%0d weight_sum=%0d total_updates=%0d",
                 subset_name, train_image_count, test_image_count, accuracy_permille,
                 avg_latency, avg_output_x1000, debug_weight_min, debug_weight_max,
                 debug_weight_sum, total_weight_updates);

        if (accuracy_permille < 100) begin
            $display("[FAIL] RTL classifier accuracy below chance-level guardrail");
            $finish(1);
        end

        $display("[PASS] RTL classifier processed train/assignment/test workflow");
        $display("*** MNIST CLASSIFIER RTL TEST PASSED ***");
        $finish(0);
    end

    initial begin
        #2000000000;
        $display("[ERROR] RTL MNIST classifier TB timed out");
        $finish(2);
    end

endmodule
