/**
 * Image byte-level utilities for pi-vision-bridge.
 *
 * - parseImageDimensions: cheap header sniffing (PNG/JPEG/GIF/WebP) with no
 *   model call and no re-encode, so the swap text can publish an image's
 *   pixel dimensions. Unparseable formats degrade silently — no dimensions
 *   line, no failure.
 * - cropImage: cuts out a [x, y, w, h] pixel region for describe_image's
 *   region zoom. PNG is cropped natively (decode → crop → re-encode, pure
 *   Node zlib — the image path must never *require* ffmpeg); every other
 *   format falls back to ffmpeg when it is on PATH (video users have it).
 *   Crop artifacts are temporary files that are always cleaned up.
 */

import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deflateSync, inflateSync } from "node:zlib";
import type { ImageContent } from "@earendil-works/pi-ai";
import { debug } from "./config.ts";

export interface ImageDimensions {
	width: number;
	height: number;
}

export interface Region {
	x: number;
	y: number;
	w: number;
	h: number;
}

const PNG_SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

const isPng = (b: Buffer): boolean => b.length >= 24 && b.subarray(0, 8).equals(PNG_SIG);

// ---------------------------------------------------------------------------
// Dimensions (header-only)
// ---------------------------------------------------------------------------

/** Parse pixel dimensions from an image header; undefined when not cheaply parseable. */
export function parseImageDimensions(data: Buffer): ImageDimensions | undefined {
	if (isPng(data)) return parsePngSize(data);
	if (data.length >= 4 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) return parseJpegSize(data);
	if (data.length >= 10 && data.toString("ascii", 0, 3) === "GIF") {
		return { width: data.readUInt16LE(6), height: data.readUInt16LE(8) };
	}
	if (
		data.length >= 30 &&
		data.toString("ascii", 0, 4) === "RIFF" &&
		data.toString("ascii", 8, 12) === "WEBP"
	) {
		return parseWebpSize(data);
	}
	return undefined;
}

function parsePngSize(data: Buffer): ImageDimensions | undefined {
	// First chunk must be IHDR: sig(8) + len(4) + "IHDR"(4) + width(4) + height(4).
	if (data.toString("ascii", 12, 16) !== "IHDR") return undefined;
	const width = data.readUInt32BE(16);
	const height = data.readUInt32BE(20);
	return width > 0 && height > 0 ? { width, height } : undefined;
}

function parseJpegSize(data: Buffer): ImageDimensions | undefined {
	// Walk segments after SOI; the first SOF marker carries the frame size.
	let pos = 2;
	while (pos + 9 < data.length) {
		if (data[pos] !== 0xff) {
			pos++;
			continue;
		}
		const marker = data[pos + 1];
		if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
			pos += 2; // standalone markers: no length payload
			continue;
		}
		const length = data.readUInt16BE(pos + 2);
		const isSof =
			marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
		if (isSof) {
			const height = data.readUInt16BE(pos + 5);
			const width = data.readUInt16BE(pos + 7);
			return width > 0 && height > 0 ? { width, height } : undefined;
		}
		if (length < 2) return undefined; // corrupt length; bail
		pos += 2 + length;
	}
	return undefined;
}

function parseWebpSize(data: Buffer): ImageDimensions | undefined {
	// Chunks after "RIFF<size>WEBP": fourcc + LE32 size + payload.
	let off = 12;
	while (off + 8 <= data.length) {
		const fourcc = data.toString("ascii", off, off + 4);
		const size = data.readUInt32LE(off + 4);
		const payload = off + 8;
		if (fourcc === "VP8X" && payload + 10 <= data.length) {
			// 4 bytes flags, then 3-byte LE canvas w-1 and h-1.
			const width = 1 + (data[payload + 4] | (data[payload + 5] << 8) | (data[payload + 6] << 16));
			const height = 1 + (data[payload + 7] | (data[payload + 8] << 8) | (data[payload + 9] << 16));
			return { width, height };
		}
		if (fourcc === "VP8L" && payload + 5 <= data.length && data[payload] === 0x2f) {
			// 1-byte signature, then LSB-first 14-bit w-1 and 14-bit h-1.
			const b1 = data[payload + 1];
			const b2 = data[payload + 2];
			const b3 = data[payload + 3];
			const b4 = data[payload + 4];
			const width = 1 + (b1 | ((b2 & 0x3f) << 8));
			const height = 1 + (((b2 >> 6) & 0x03) | (b3 << 2) | ((b4 & 0x0f) << 10));
			return { width, height };
		}
		off = payload + size + (size % 2); // chunks are padded to even sizes
	}
	return undefined;
}

// ---------------------------------------------------------------------------
// Region clamping
// ---------------------------------------------------------------------------

/**
 * Clamp a [x, y, w, h] box to the image bounds. Origins are pulled inside,
 * extents are truncated. The caller decides what a zero-area result means.
 */
export function clampRegion(
	box: readonly number[],
	dims: ImageDimensions,
): { region: Region; clamped: boolean } {
	const [ox, oy, ow, oh] = box;
	const x = Math.max(0, Math.min(ox, dims.width));
	const y = Math.max(0, Math.min(oy, dims.height));
	const w = Math.min(ow, dims.width - x);
	const h = Math.min(oh, dims.height - y);
	return { region: { x, y, w, h }, clamped: x !== ox || y !== oy || w !== ow || h !== oh };
}

// ---------------------------------------------------------------------------
// Cropping
// ---------------------------------------------------------------------------

/**
 * Crop a region out of an image, returning a standalone image (PNG).
 * Full-image regions pass the original content through untouched.
 * Injectable seam for tests: setCropperForTests.
 */
export function cropImage(
	image: ImageContent,
	region: Region,
	options?: { signal?: AbortSignal },
): Promise<ImageContent> {
	return cropImageImpl(image, region, options);
}

type Cropper = typeof defaultCropImage;
let cropImageImpl: Cropper = defaultCropImage;

/** Replace/restore the crop implementation (harness seam; stubs the encode step). */
export function setCropperForTests(impl: Cropper | undefined): void {
	cropImageImpl = impl ?? defaultCropImage;
}

async function defaultCropImage(
	image: ImageContent,
	region: Region,
	options?: { signal?: AbortSignal },
): Promise<ImageContent> {
	const bytes = Buffer.from(image.data, "base64");
	if (isPng(bytes)) {
		try {
			const png = decodePng(bytes);
			if (region.x === 0 && region.y === 0 && region.w === png.width && region.h === png.height) {
				return image; // nothing to cut out
			}
			const cropped = encodePng(cropPng(png, region));
			return { type: "image", data: cropped.toString("base64"), mimeType: "image/png" };
		} catch (err) {
			// Unsupported PNG variant (16-bit, palette bit-depth, interlaced):
			// fall through to the ffmpeg cropper instead of failing outright.
			debug("native PNG crop unsupported, trying ffmpeg:", err instanceof Error ? err.message : err);
		}
	}
	return cropWithFfmpeg(bytes, region, image.mimeType, options);
}

/** Crop via ffmpeg into a temp PNG; used for formats without a native cropper. */
async function cropWithFfmpeg(
	bytes: Buffer,
	region: Region,
	mimeType: string,
	options?: { signal?: AbortSignal },
): Promise<ImageContent> {
	const format = mimeType.split("/")[1]?.split("+")[0]?.replace(/[^a-z0-9]/gi, "") || "bin";
	const dir = mkdtempSync(join(tmpdir(), "pi-vision-bridge-crop-"));
	try {
		const input = join(dir, `input.${format}`);
		const output = join(dir, "crop.png");
		writeFileSync(input, bytes);
		const res = await runFfmpeg(
			[
				"-hide_banner", "-loglevel", "error",
				"-y",
				"-i", input,
				"-vf", `crop=${region.w}:${region.h}:${region.x}:${region.y}`,
				"-frames:v", "1",
				output,
			],
			options?.signal,
		);
		if (res.code !== 0) {
			throw new Error(`ffmpeg crop failed: ${res.stderr.trim().split("\n").slice(-1)[0] ?? "unknown error"}`);
		}
		const cropped = readFileSync(output);
		if (!isPng(cropped)) throw new Error("ffmpeg produced no PNG output");
		return { type: "image", data: cropped.toString("base64"), mimeType: "image/png" };
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

function runFfmpeg(args: string[], signal?: AbortSignal): Promise<{ code: number; stderr: string }> {
	return new Promise((resolve, reject) => {
		const child = spawn("ffmpeg", args, { windowsHide: true });
		let stderr = "";
		const timer = setTimeout(() => {
			child.kill();
			reject(new Error("ffmpeg timed out after 30s"));
		}, 30_000);
		const onAbort = () => child.kill();
		signal?.addEventListener("abort", onAbort, { once: true });
		const done = () => {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		};
		child.stderr.on("data", (d) => (stderr += d.toString()));
		child.on("error", (err) => {
			done();
			reject(err); // ENOENT: ffmpeg not on PATH
		});
		child.on("close", (code) => {
			done();
			if (signal?.aborted) {
				reject(new Error("aborted"));
				return;
			}
			resolve({ code: code ?? -1, stderr });
		});
	});
}

// ---------------------------------------------------------------------------
// Minimal PNG codec (8-bit, non-interlaced, color types 0/2/3/4/6)
// ---------------------------------------------------------------------------

const COLOR_CHANNELS: Record<number, number> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

interface PngImage {
	width: number;
	height: number;
	colorType: number;
	channels: number;
	/** Defiltered pixel data: width * height * channels bytes. */
	raw: Buffer;
	/** Palette + transparency chunks, kept verbatim for color type 3. */
	plte?: Buffer;
	trns?: Buffer;
}

function decodePng(buf: Buffer): PngImage {
	let width = 0;
	let height = 0;
	let bitDepth = 0;
	let colorType = 0;
	let interlace = 0;
	let plte: Buffer | undefined;
	let trns: Buffer | undefined;
	const idat: Buffer[] = [];

	let off = 8;
	while (off + 8 <= buf.length) {
		const len = buf.readUInt32BE(off);
		const type = buf.toString("ascii", off + 4, off + 8);
		const start = off + 8;
		const end = start + len;
		if (end > buf.length) break; // truncated
		if (type === "IHDR") {
			width = buf.readUInt32BE(start);
			height = buf.readUInt32BE(start + 4);
			bitDepth = buf[start + 8];
			colorType = buf[start + 9];
			interlace = buf[start + 12];
		} else if (type === "PLTE") {
			plte = Buffer.from(buf.subarray(start, end));
		} else if (type === "tRNS") {
			trns = Buffer.from(buf.subarray(start, end));
		} else if (type === "IDAT") {
			idat.push(buf.subarray(start, end));
		} else if (type === "IEND") {
			break;
		}
		off = end + 4; // skip CRC
	}

	if (!width || !height) throw new Error("PNG has no IHDR");
	if (bitDepth !== 8) throw new Error(`unsupported PNG bit depth ${bitDepth}`);
	if (interlace !== 0) throw new Error("interlaced PNG is not supported");
	const channels = COLOR_CHANNELS[colorType];
	if (!channels) throw new Error(`unsupported PNG color type ${colorType}`);
	if (colorType === 3 && !plte) throw new Error("palette PNG without PLTE");

	const stride = width * channels;
	const inflated = inflateSync(Buffer.concat(idat));
	const expected = (stride + 1) * height; // filter byte per scanline
	if (inflated.length < expected) throw new Error("truncated PNG pixel data");

	const raw = Buffer.alloc(width * height * channels);
	let prev = Buffer.alloc(stride); // zero row above the first
	for (let y = 0; y < height; y++) {
		const rowStart = y * (stride + 1);
		const filter = inflated[rowStart];
		const line = inflated.subarray(rowStart + 1, rowStart + 1 + stride);
		const cur = raw.subarray(y * stride, (y + 1) * stride);
		defilterRow(filter, line, prev, cur, channels);
		prev = cur;
	}
	return { width, height, colorType, channels, raw, plte, trns };
}

function defilterRow(filter: number, line: Buffer, prev: Buffer, out: Buffer, bpp: number): void {
	if (filter === 0) {
		line.copy(out);
		return;
	}
	for (let i = 0; i < line.length; i++) {
		const value = line[i];
		const left = i >= bpp ? out[i - bpp] : 0;
		const up = prev[i];
		const upLeft = i >= bpp ? prev[i - bpp] : 0;
		switch (filter) {
			case 1:
				out[i] = (value + left) & 0xff;
				break;
			case 2:
				out[i] = (value + up) & 0xff;
				break;
			case 3:
				out[i] = (value + ((left + up) >> 1)) & 0xff;
				break;
			case 4:
				out[i] = (value + paeth(left, up, upLeft)) & 0xff;
				break;
			default:
				throw new Error(`unknown PNG filter type ${filter}`);
		}
	}
}

function paeth(a: number, b: number, c: number): number {
	const p = a + b - c;
	const pa = Math.abs(p - a);
	const pb = Math.abs(p - b);
	const pc = Math.abs(p - c);
	return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

function cropPng(png: PngImage, region: Region): PngImage {
	const { channels } = png;
	const srcStride = png.width * channels;
	const dstStride = region.w * channels;
	const raw = Buffer.alloc(region.w * region.h * channels);
	for (let y = 0; y < region.h; y++) {
		const srcStart = (region.y + y) * srcStride + region.x * channels;
		png.raw.copy(raw, y * dstStride, srcStart, srcStart + dstStride);
	}
	return { ...png, width: region.w, height: region.h, raw };
}

function encodePng(png: PngImage): Buffer {
	const chunks: Buffer[] = [];
	const ihdr = Buffer.alloc(13);
	ihdr.writeUInt32BE(png.width, 0);
	ihdr.writeUInt32BE(png.height, 4);
	ihdr[8] = 8; // bit depth
	ihdr[9] = png.colorType;
	ihdr[10] = 0; // compression: deflate
	ihdr[11] = 0; // filter method
	ihdr[12] = 0; // interlace: none
	chunks.push(pngChunk("IHDR", ihdr));
	if (png.colorType === 3 && png.plte) {
		chunks.push(pngChunk("PLTE", png.plte));
		if (png.trns) chunks.push(pngChunk("tRNS", png.trns));
	}
	const stride = png.width * png.channels;
	const scan = Buffer.alloc((stride + 1) * png.height);
	for (let y = 0; y < png.height; y++) {
		scan[y * (stride + 1)] = 0; // filter: none
		png.raw.copy(scan, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
	}
	chunks.push(pngChunk("IDAT", deflateSync(scan, { level: 6 })));
	chunks.push(pngChunk("IEND", Buffer.alloc(0)));
	return Buffer.concat([PNG_SIG, ...chunks]);
}

function pngChunk(type: string, data: Buffer): Buffer {
	const out = Buffer.alloc(12 + data.length);
	out.writeUInt32BE(data.length, 0);
	out.write(type, 4, "ascii");
	data.copy(out, 8);
	out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
	return out;
}

const CRC_TABLE = (() => {
	const table = new Uint32Array(256);
	for (let n = 0; n < 256; n++) {
		let c = n;
		for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
		table[n] = c >>> 0;
	}
	return table;
})();

function crc32(buf: Buffer): number {
	let crc = 0xffffffff;
	for (let i = 0; i < buf.length; i++) crc = CRC_TABLE[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
	return (crc ^ 0xffffffff) >>> 0;
}
