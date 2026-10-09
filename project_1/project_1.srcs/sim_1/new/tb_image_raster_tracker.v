`timescale 1ns / 1ps

module tb_image_raster_tracker;

    reg clk;
    reg rst_n;

    reg [15:0] cfg_width;
    reg [15:0] cfg_height;

    reg s_valid;
    reg s_ready;
    reg s_sof;
    reg s_eol;

    wire [15:0] pixel_x;
    wire [15:0] pixel_y;
    wire        coord_valid;
    wire        frame_done;
    wire        format_error;

    integer x;
    integer y;
    integer errors;
    integer done_count;

    image_raster_tracker dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .cfg_width    (cfg_width),
        .cfg_height   (cfg_height),
        .s_valid      (s_valid),
        .s_ready      (s_ready),
        .s_sof        (s_sof),
        .s_eol        (s_eol),
        .pixel_x      (pixel_x),
        .pixel_y      (pixel_y),
        .coord_valid  (coord_valid),
        .frame_done   (frame_done),
        .format_error (format_error)
    );

    // 10 ns 周期的仿真时钟
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // 检查当前输入像素对应的坐标和帧结束标记
    task check_current_pixel;
        input [15:0] expected_x;
        input [15:0] expected_y;
        input        expected_frame_done;
        begin
            #1;

            if ((coord_valid !== 1'b1) ||
                (pixel_x !== expected_x) ||
                (pixel_y !== expected_y) ||
                (frame_done !== expected_frame_done) ||
                (format_error !== 1'b0)) begin

                errors = errors + 1;

                $display(
                    "FAIL: expected x=%0d y=%0d frame_done=%b; got x=%0d y=%0d frame_done=%b format_error=%b",
                    expected_x,
                    expected_y,
                    expected_frame_done,
                    pixel_x,
                    pixel_y,
                    frame_done,
                    format_error
                );
            end else begin
                $display(
                    "PASS: x=%0d y=%0d frame_done=%b",
                    pixel_x,
                    pixel_y,
                    frame_done
                );
            end

            if (frame_done)
                done_count = done_count + 1;
        end
    endtask

    initial begin
        rst_n       = 1'b0;
        cfg_width   = 16'd4;
        cfg_height  = 16'd3;
        s_valid     = 1'b0;
        s_ready     = 1'b0;
        s_sof       = 1'b0;
        s_eol       = 1'b0;
        errors      = 0;
        done_count  = 0;

        // 保持复位两个时钟周期
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // 发送 4 列 × 3 行的图像
        for (y = 0; y < 3; y = y + 1) begin
            for (x = 0; x < 4; x = x + 1) begin

                @(negedge clk);
                s_valid = 1'b1;
                s_sof   = ((x == 0) && (y == 0));
                s_eol   = (x == 3);

                // 第二颗像素 x=1,y=0 暂停两拍，模拟下游背压
                if ((x == 1) && (y == 0)) begin
                    s_ready = 1'b0;

                    // ready=0 时像素没有被接收，坐标应保持在 (1,0)
                    check_current_pixel(16'd1, 16'd0, 1'b0);

                    repeat (2) begin
                        @(posedge clk);
                        check_current_pixel(16'd1, 16'd0, 1'b0);
                    end

                    // 下游恢复接收，当前这颗像素完成握手
                    @(negedge clk);
                    s_ready = 1'b1;
                    check_current_pixel(16'd1, 16'd0, 1'b0);

                    @(posedge clk);
                    #1;
                end else begin
                    s_ready = 1'b1;

                    check_current_pixel(
                        x,
                        y,
                        ((x == 3) && (y == 2))
                    );

                    @(posedge clk);
                    #1;
                end
            end
        end

        // 停止输入
        @(negedge clk);
        s_valid = 1'b0;
        s_ready = 1'b0;
        s_sof   = 1'b0;
        s_eol   = 1'b0;

        #1;

        if ((errors == 0) &&
            (done_count == 1) &&
            (format_error == 1'b0)) begin
            $display("TEST PASS: 4x3 image stream checked successfully.");
        end else begin
            $display(
                "TEST FAIL: errors=%0d frame_done_count=%0d format_error=%b",
                errors,
                done_count,
                format_error
            );
        end

        $finish;
    end

endmodule