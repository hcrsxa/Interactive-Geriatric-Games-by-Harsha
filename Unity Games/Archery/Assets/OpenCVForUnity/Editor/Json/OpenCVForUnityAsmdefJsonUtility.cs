#if UNITY_6000_0_OR_NEWER
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEditor;

namespace OpenCVForUnity.Editor
{
    /// <summary>
    /// Utility for reading and editing Assembly Definition (.asmdef) JSON as a <see cref="JObject"/> (Newtonsoft.Json).
    /// For Unity Sentis integration, only <c>references</c> and <c>versionDefines</c> are modified; other properties are preserved.
    /// </summary>
    internal static class OpenCVForUnityAsmdefJsonUtility
    {
        public static bool TryParseAsmdef(string fullPath, out JObject root, out string error)
        {
            root = null;
            error = null;

            if (!File.Exists(fullPath))
            {
                error = "File not found: " + fullPath;
                return false;
            }

            try
            {
                string text = File.ReadAllText(fullPath);
                root = JObject.Parse(text);
                return true;
            }
            catch (Exception ex)
            {
                error = fullPath + " : " + ex.Message;
                return false;
            }
        }

        public static bool HasSentisReference(JObject root, string assemblyName)
        {
            return GetStringTokens(root["references"]).Any(s => s == assemblyName);
        }

        public static bool HasSentisVersionDefine(JObject root, string packageName, string defineSymbol)
        {
            JArray defs = root["versionDefines"] as JArray;
            if (defs == null)
            {
                return false;
            }

            foreach (JToken t in defs)
            {
                if (t is not JObject o)
                {
                    continue;
                }

                if ((string)o["name"] == packageName && (string)o["define"] == defineSymbol)
                {
                    return true;
                }
            }

            return false;
        }

        public static void ApplyIntegration(
            JObject root,
            bool enable,
            string assemblyName,
            string packageName,
            string versionExpression,
            string defineSymbol)
        {
            if (enable)
            {
                JArray refs = EnsureArray(root, "references");
                if (!GetStringTokens(refs).Any(s => s == assemblyName))
                {
                    refs.Add(assemblyName);
                }

                if (!HasSentisVersionDefine(root, packageName, defineSymbol))
                {
                    JArray defs = EnsureArray(root, "versionDefines");
                    defs.Add(
                        new JObject
                        {
                            ["name"] = packageName,
                            ["expression"] = versionExpression,
                            ["define"] = defineSymbol
                        });
                }
            }
            else
            {
                RemoveReference(root, assemblyName);
                RemoveVersionDefines(root, packageName, defineSymbol);
            }
        }

        public static bool TryWriteAsmdef(string fullPath, JObject root, string assetPath, out string error)
        {
            error = null;

            try
            {
                string json = root.ToString(Formatting.Indented);
                File.WriteAllText(fullPath, json);
                AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
                return true;
            }
            catch (Exception ex)
            {
                error = fullPath + " : " + ex.Message;
                return false;
            }
        }

        private static IEnumerable<string> GetStringTokens(JToken token)
        {
            if (token == null || token.Type != JTokenType.Array)
            {
                yield break;
            }

            foreach (JToken item in (JArray)token)
            {
                if (item != null && item.Type == JTokenType.String)
                {
                    yield return item.Value<string>();
                }
            }
        }

        private static JArray EnsureArray(JObject root, string key)
        {
            JArray arr = root[key] as JArray;
            if (arr == null)
            {
                arr = new JArray();
                root[key] = arr;
            }

            return arr;
        }

        private static void RemoveReference(JObject root, string assemblyName)
        {
            JArray refs = root["references"] as JArray;
            if (refs == null)
            {
                return;
            }

            JArray next = new JArray();
            foreach (JToken t in refs)
            {
                if (t != null && t.Type == JTokenType.String && (string)t == assemblyName)
                {
                    continue;
                }

                next.Add(t);
            }

            root["references"] = next;
        }

        private static void RemoveVersionDefines(JObject root, string packageName, string defineSymbol)
        {
            JArray defs = root["versionDefines"] as JArray;
            if (defs == null)
            {
                return;
            }

            JArray next = new JArray();
            foreach (JToken t in defs)
            {
                if (t is JObject o && (string)o["name"] == packageName && (string)o["define"] == defineSymbol)
                {
                    continue;
                }

                next.Add(t);
            }

            root["versionDefines"] = next;
        }
    }
}
#endif
