/**
 * AES-GCM encryption for BYO API keys, using the Web Crypto API
 * (available in the Convex action runtime — no Node required).
 *
 * Keys are encrypted with a server-only secret (KEY_ENCRYPTION_SECRET) and the
 * ciphertext is the only thing persisted. Must be called from an action
 * (randomness is non-deterministic, so not allowed in queries/mutations).
 */

function getSecret(): string {
  const secret = process.env.KEY_ENCRYPTION_SECRET;
  if (!secret || secret.length < 16) {
    throw new Error(
      "KEY_ENCRYPTION_SECRET is not set (need >=16 chars). Set it with `npx convex env set KEY_ENCRYPTION_SECRET <value>`.",
    );
  }
  return secret;
}

async function deriveKey(): Promise<CryptoKey> {
  const enc = new TextEncoder().encode(getSecret());
  const hash = await crypto.subtle.digest("SHA-256", enc);
  return crypto.subtle.importKey("raw", hash, { name: "AES-GCM" }, false, [
    "encrypt",
    "decrypt",
  ]);
}

function toBase64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function fromBase64(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function encryptSecret(plaintext: string): Promise<string> {
  const key = await deriveKey();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ct = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    new TextEncoder().encode(plaintext),
  );
  const ctBytes = new Uint8Array(ct);
  const packed = new Uint8Array(iv.length + ctBytes.length);
  packed.set(iv, 0);
  packed.set(ctBytes, iv.length);
  return toBase64(packed);
}

export async function decryptSecret(packedB64: string): Promise<string> {
  const key = await deriveKey();
  const packed = fromBase64(packedB64);
  const iv = packed.slice(0, 12);
  const ct = packed.slice(12);
  const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
  return new TextDecoder().decode(pt);
}
