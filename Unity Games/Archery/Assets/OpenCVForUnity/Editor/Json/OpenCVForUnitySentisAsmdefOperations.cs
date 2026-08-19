#if UNITY_6000_0_OR_NEWER
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEngine;

namespace OpenCVForUnity.Editor
{
    /// <summary>
    /// Sentis asmdef JSON editing (Newtonsoft.Json). Lives in <c>EnoxSoftware.OpenCVForUnity.Editor.Json</c>; invoked from the main Editor assembly via reflection.
    /// </summary>
    public static class OpenCVForUnitySentisAsmdefOperations
    {
        private const string SentisAssemblyName = "Unity.InferenceEngine";

        private const string SentisPackageName = "com.unity.ai.inference";

        private const string SentisVersionExpression = "2.6.1";

        private const string SentisDefineSymbol = "OPENCV_SENTIS_AVAILABLE";

        private static string GetSentisMainAsmdefAssetPath(string openCvForUnityRoot)
        {
            return openCvForUnityRoot + "/EnoxSoftware.OpenCVForUnity.asmdef";
        }

        private static string GetSentisExampleAsmdefAssetPath(string openCvForUnityRoot)
        {
            return openCvForUnityRoot + "/Examples/EnoxSoftware.OpenCVForUnityExample.asmdef";
        }

        private static string GetSentisEditorAsmdefAssetPath(string openCvForUnityRoot)
        {
            return openCvForUnityRoot + "/Editor/EnoxSoftware.OpenCVForUnity.Editor.asmdef";
        }

        private static string AssetPathToFullPath(string assetPath)
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            return Path.GetFullPath(Path.Combine(projectRoot, assetPath.Replace('/', Path.DirectorySeparatorChar)));
        }

        /// <summary>
        /// Verifies the main and Editor asmdefs include the Unity Sentis assembly reference and matching versionDefines.
        /// The Example asmdef is not validated when its file is missing; when the file exists, it is validated the same way as the main asmdef.
        /// </summary>
        public static bool ValidateSentisAsmdefIntegration(string openCvForUnityRoot)
        {
            string mainAssetPath = GetSentisMainAsmdefAssetPath(openCvForUnityRoot);
            if (!OpenCVForUnityAsmdefJsonUtility.TryParseAsmdef(AssetPathToFullPath(mainAssetPath), out JObject mainData, out _))
            {
                return false;
            }

            if (!OpenCVForUnityAsmdefJsonUtility.HasSentisReference(mainData, SentisAssemblyName)
                || !OpenCVForUnityAsmdefJsonUtility.HasSentisVersionDefine(mainData, SentisPackageName, SentisDefineSymbol))
            {
                return false;
            }

            string editorAssetPath = GetSentisEditorAsmdefAssetPath(openCvForUnityRoot);
            if (!OpenCVForUnityAsmdefJsonUtility.TryParseAsmdef(AssetPathToFullPath(editorAssetPath), out JObject editorData, out _))
            {
                return false;
            }

            if (!OpenCVForUnityAsmdefJsonUtility.HasSentisReference(editorData, SentisAssemblyName)
                || !OpenCVForUnityAsmdefJsonUtility.HasSentisVersionDefine(editorData, SentisPackageName, SentisDefineSymbol))
            {
                return false;
            }

            string exampleAssetPath = GetSentisExampleAsmdefAssetPath(openCvForUnityRoot);
            string exampleFullPath = AssetPathToFullPath(exampleAssetPath);
            if (!File.Exists(exampleFullPath))
            {
                return true;
            }

            if (!OpenCVForUnityAsmdefJsonUtility.TryParseAsmdef(exampleFullPath, out JObject exampleData, out _))
            {
                return false;
            }

            if (!OpenCVForUnityAsmdefJsonUtility.HasSentisReference(exampleData, SentisAssemblyName)
                || !OpenCVForUnityAsmdefJsonUtility.HasSentisVersionDefine(exampleData, SentisPackageName, SentisDefineSymbol))
            {
                return false;
            }

            return true;
        }

        /// <summary>
        /// Writes Sentis integration state to OpenCV for Unity asmdef files under <paramref name="openCvForUnityRoot"/>.
        /// </summary>
        public static void ApplySentisAsmdefIntegration(string openCvForUnityRoot, bool enable)
        {
            var errors = new List<string>();

            string mainAssetPath = GetSentisMainAsmdefAssetPath(openCvForUnityRoot);
            string mainFullPath = AssetPathToFullPath(mainAssetPath);
            if (!OpenCVForUnityAsmdefJsonUtility.TryParseAsmdef(mainFullPath, out JObject mainData, out string mainReadError))
            {
                errors.Add(mainReadError);
            }
            else
            {
                OpenCVForUnityAsmdefJsonUtility.ApplyIntegration(
                    mainData,
                    enable,
                    SentisAssemblyName,
                    SentisPackageName,
                    SentisVersionExpression,
                    SentisDefineSymbol);

                if (!OpenCVForUnityAsmdefJsonUtility.TryWriteAsmdef(mainFullPath, mainData, mainAssetPath, out string mainWriteError))
                {
                    errors.Add(mainWriteError);
                }
            }

            string editorAssetPath = GetSentisEditorAsmdefAssetPath(openCvForUnityRoot);
            string editorFullPath = AssetPathToFullPath(editorAssetPath);
            if (!OpenCVForUnityAsmdefJsonUtility.TryParseAsmdef(editorFullPath, out JObject editorData, out string editorReadError))
            {
                errors.Add(editorReadError);
            }
            else
            {
                OpenCVForUnityAsmdefJsonUtility.ApplyIntegration(
                    editorData,
                    enable,
                    SentisAssemblyName,
                    SentisPackageName,
                    SentisVersionExpression,
                    SentisDefineSymbol);

                if (!OpenCVForUnityAsmdefJsonUtility.TryWriteAsmdef(editorFullPath, editorData, editorAssetPath, out string editorWriteError))
                {
                    errors.Add(editorWriteError);
                }
            }

            string exampleAssetPath = GetSentisExampleAsmdefAssetPath(openCvForUnityRoot);
            string exampleFullPath = AssetPathToFullPath(exampleAssetPath);
            if (!File.Exists(exampleFullPath))
            {
                Debug.Log($"{nameof(OpenCVForUnitySentisAsmdefOperations)}: Optional Example asmdef not found, skipped: {exampleAssetPath}");
            }
            else
            {
                if (!OpenCVForUnityAsmdefJsonUtility.TryParseAsmdef(exampleFullPath, out JObject exampleData, out string exampleReadError))
                {
                    errors.Add(exampleReadError);
                }
                else
                {
                    OpenCVForUnityAsmdefJsonUtility.ApplyIntegration(
                        exampleData,
                        enable,
                        SentisAssemblyName,
                        SentisPackageName,
                        SentisVersionExpression,
                        SentisDefineSymbol);

                    if (!OpenCVForUnityAsmdefJsonUtility.TryWriteAsmdef(exampleFullPath, exampleData, exampleAssetPath, out string exampleWriteError))
                    {
                        errors.Add(exampleWriteError);
                    }
                }
            }

            if (errors.Count > 0)
            {
                Debug.LogError("Sentis asmdef update failed:\n" + string.Join("\n", errors));
                EditorUtility.DisplayDialog("Error", string.Join("\n", errors), "OK");
            }
            else
            {
                string logMessage = enable
                    ? "Unity Sentis inference is enabled for OpenCV for Unity Dnn example scenes (asmdef files updated)."
                    : "Unity Sentis inference is disabled for OpenCV for Unity Dnn example scenes (asmdef files updated).";
                Debug.Log(logMessage);
                EditorUtility.DisplayDialog("Success", logMessage, "OK");
            }
        }
    }
}
#endif
