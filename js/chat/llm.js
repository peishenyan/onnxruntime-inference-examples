import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = document.location.pathname.replace('index.html', '') + 'dist/';


function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

//
// load file from server or cache
//
async function fetchAndCache(url) {
    try {
        // const cache = await caches.open("onnx");
        // let cachedResponse = await cache.match(url);
        // if (cachedResponse === undefined) {
        if (true) {
            log(`${url} (network)`);
            const buffer = await (await fetch(url)).arrayBuffer();
            // try {
            //     await cache.put(url, new Response(buffer));
            // } catch (error) {
            //     console.error(error);
            // }
            return buffer;
        }
        log(`${url} (cached)`);
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`can't fetch ${url}`);
        throw error;
    }
}

//
// class to handle a large language model on top of onnxruntime-web
//
export class LLM {
    sess = undefined;
    profiler = false;
    feed = {};
    output_tokens = [];
    eos = 2;
    need_position_ids = true;
    stop = false;
    kv_dims = [];
    dtype = "float16";
    max_tokens = 128;

    constructor() {
    }

    async load(model, options, flag=true) {
        const provider = options.provider || "webgpu";
        const verbose = options.verbose;
        const local = options.local;
        const hasFP16 = (provider === "wasm") ? false : options.hasFP16;
        this.profiler = options.profiler;
        const max_tokens = options.max_tokens;

        const model_path ="models/";
        let model_file_1 = "Phi_3_mini_4k_instruct_decoder_static_non_kvcache_lm_ex_v21_new_INT4_QDQ_padded.onnx";
        let model_file_2 = "Phi_3_mini_4k_instruct_decoder_static_kvcache_128_lm_ex_v21_new_INT4_QDQ_padded.onnx";
        
        log(`loading... ${model.name},  ${provider}`);

        const model_bytes_1 = await fetchAndCache(model_path + model_file_1);
        // const externaldata_1 = (model.externaldata) ? await fetchAndCache(model_path + model_file_1 + '.data') : false;
        const externaldata_1 = model_path + model_file_1 + '.data'
        // const model_bytes_2 = await fetchAndCache(model_path + model_file_2);
        // const externaldata_2 = (model.externaldata) ? await fetchAndCache(model_path + model_file_2 + '.data') : false; 
        let modelSize_1 = model_bytes_1.byteLength;
        // let modelSize_2 = model_bytes_2.byteLength;
        if (externaldata_1) {
            modelSize_1 += externaldata_1.byteLength;
        }
        // if (externaldata_2) {
        //     modelSize_2 += externaldata_2.byteLength;
        // }
        log(`model 1 size ${Math.round(modelSize_1 / 1024 / 1024)} MB`);
        // log(`model 2 size ${Math.round(modelSize_2 / 1024 / 1024)} MB`);

        const opt_1 = {
            executionProviders: [provider],
            preferredOutputLocation: {},
        }
        // const opt_2 = {
        //     executionProviders: [provider],
        //     preferredOutputLocation: {},
        // }

        switch (provider) {
            case "webgpu":
                for (let i = 0; i < 32; ++i) {
                    opt_1.preferredOutputLocation[`present.${i}.key`] = 'gpu-buffer';
                    opt_1.preferredOutputLocation[`present.${i}.value`] = 'gpu-buffer';
                    // opt_2.preferredOutputLocation[`present.${i}.key`] = 'gpu-buffer';
                    // opt_2.preferredOutputLocation[`present.${i}.value`] = 'gpu-buffer';
                }
                break;
        }

        if (externaldata_1 !== undefined) {
            opt_1.externaldata = [
                {
                    data: externaldata_1,
                    path: model_file_1 + ".data",
                },
            ]
        }
        // if (externaldata_2 !== undefined) {
        //     opt_2.externaldata = [
        //         {
        //             data: externaldata_2,
        //             path: model_file_2 + ".data",
        //         },
        //     ]
        // }

        if (verbose) {
            opt_1.logSeverityLevel = 0;
            opt_1.logVerbosityLevel = 0;
            opt_1.env.logLevel = "verbose";
            // opt_2.logSeverityLevel = 0;
            // opt_2.logVerbosityLevel = 0;
            // opt_2.env.logLevel = "verbose";
        }

        ort.env.webgpu.profiling = {}
        if (this.profiler) {
            opt_1.enableProfiling = true;
            // opt_2.enableProfiling = true;
            ort.env.webgpu.profilingMode = 'default';
            ort.env.webgpu.profiling.mode = 'default';
        }

        this.sess_1 = await ort.InferenceSession.create(model_bytes_1, opt_1);
        // this.sess_2 = await ort.InferenceSession.create(model_bytes_2, opt_2);
        this.sess_2 = undefined;
        this.eos = 50256;
        this.kv_dims = [1, 32, 2*max_tokens-1, 96];
        this.dtype = (hasFP16) ? "float16" : "float32";
        this.num_layers = 32;
        if (!flag) {
            this.initilize_feed();
        }
    }

    initilize_feed() {
        const feed = this.feed;

        // dispose of previous gpu buffers
        for (const name in feed) {
            const t = feed[name];
            if (t.location === 'gpu-buffer') {
                t.dispose();
            }
        }
        this.feed = {};
        // key value cache is zero copy, just pass gpu buffer as referece
        const empty = (this.dtype === "float16") ? new Uint16Array() : [];
        for (let i = 0; i < this.num_layers; ++i) {
            this.feed[`past_key_values.${i}.key`] = new ort.Tensor(this.dtype, empty, this.kv_dims)
            this.feed[`past_key_values.${i}.value`] = new ort.Tensor(this.dtype, empty, this.kv_dims)
        }
        this.output_tokens = [];
    }

    //
    // poor mens argmax
    argmax(t) {
        const arr = t.data;
        const start = t.dims[2] * (t.dims[1] - 1);
        let max = arr[start];
        let maxidx = 0;

        for (let i = 0; i < t.dims[2]; i++) {
            const val = arr[i + start];
            if (!isFinite(val)) {
                throw new Error("found infinitive in logits");
            }
            if (val > max) {
                max = arr[i + start];
                maxidx = i;
            }
        }
        return maxidx;
    }

    //
    // update key value cache
    //
    update_kv_cache(feed, outputs) {
        for (const name in outputs) {
            if (name.startsWith('present')) {
                let newName = name.replace('present', 'past_key_values');
                // dispose previous gpu buffers
                const t = feed[newName];
                if (t.location === 'gpu-buffer') {
                    t.dispose();
                }
                feed[newName] = outputs[name];
            }
        }
    }

    //
    // tell generate to stop()
    //
    abort() {
        this.stop = true;
    }

    // 
    // prefill prompt and generate tokens, greedy search only
    //
    async generate(tokens, callback, options) {
        const max_tokens = options.max_tokens || 128;
        const feed = this.feed;
        const input_len = tokens.length;

        const pad_value = 0n;
        if (tokens.length < max_tokens) {
            const padding_length = max_tokens - tokens.length;
            const padding = Array.from({ length: padding_length }, () => pad_value);
            tokens.push(...padding);
            let attn_mask = Array.from({ length: tokens.length }, () => 1n);
            attn_mask.push(...padding);
        }
        const input_ids = new ort.Tensor('int64', BigInt64Array.from(tokens.map(BigInt)), [1, max_tokens]);
        feed['input_ids'] = input_ids;
        const attention_mask = new ort.Tensor('int64', BigInt64Array.from(attn_mask.map(BigInt)), [1, max_tokens]);
        feed['attention_mask'] = attention_mask;
        this.stop = false;

        let last_token = 0n;

        const outputs_0 = await this.sess_1.run(feed);

        last_token = BigInt(this.argmax(outputs_0.logits));
        this.output_tokens.push(last_token);

        let seqlen = input_len;

        while (last_token != this.eos && last_token != 32007 && seqlen < 2*max_tokens-1 && !this.stop) {
            feed['input_ids'] = new ort.Tensor('int64', BigInt64Array.from([last_token]), [1, 1]);
            feed['attention_mask'] = new ort.Tensor('int64', BigInt64Array.from(attn_mask.map(BigInt)), [1, 2*max_tokens]);
            feed['position_ids'] = new ort.Tensor('int64', BigInt64Array.from([BigInt(seqlen)]), [1, 1]);
            
            const outputs = await this.sess_2.run(feed);
            last_token = BigInt(this.argmax(outputs.logits));
            this.output_tokens.push(last_token);
            if (callback && !this.profiler) {
                callback(this.output_tokens);
            }
            this.update_kv_cache(feed, outputs);
            seqlen += 1;
            attn_mask[seqlen] = 1;
        }
        if (this.profiler) {
            this.sess.endProfiling();
        }
        return this.output_tokens;
    }
}
